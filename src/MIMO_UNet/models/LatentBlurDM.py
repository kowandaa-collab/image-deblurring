"""
LatentBlurDM: Latent Exposure Diffusion Model for image deblurring.

Improvements over the original:
  - Cosine noise schedule (more stable than linear)
  - Sinusoidal timestep embeddings (richer time signal than scalar t/T)
  - Type annotations and docstrings throughout
  - Configurable MLP depth/width
  - Reverse step extracted as a standalone method for clarity
  - q_sample supports arbitrary timestep (enables supervised mid-output)
  - Removed dead/commented code and hardcoded paths
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from .LatentEncoder import LE_arch


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal timestep embedding (Vaswani et al., 2017)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=device).float() / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=1)


class ResMLP(nn.Module):
    """Residual MLP block with skip connection."""

    def __init__(self, n_feats: int = 512) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(n_feats, n_feats),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DenoiseHead(nn.Module):
    """Shared architecture for noise and blur-residual estimators.

    Args:
        n_feats:       Base latent feature size (must match LE_arch output / 4).
        n_res_blocks:  Number of ResMLP blocks.
        time_emb_dim:  Dimension of sinusoidal time embedding.
    """

    def __init__(
        self,
        n_feats: int = 64,
        n_res_blocks: int = 5,
        time_emb_dim: int = 64,
    ) -> None:
        super().__init__()
        self.time_emb = SinusoidalPosEmb(time_emb_dim)
        latent_dim = n_feats * 4
        in_dim = latent_dim * 2 + time_emb_dim

        layers: list[nn.Module] = [
            nn.Linear(in_dim, latent_dim),
            nn.LeakyReLU(0.1, inplace=True),
        ]
        for _ in range(n_res_blocks):
            layers.append(ResMLP(latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        t_emb = self.time_emb(t)
        h = torch.cat([c, t_emb, x], dim=1)
        return self.net(h)


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule (Nichol & Dhariwal, 2021) — smoother than linear."""
    steps = timesteps + 1
    t = torch.linspace(0.0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1.0 + s) * math.pi / 2.0) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(0.0, 0.999)


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    return torch.linspace(0.0, 0.02, timesteps)


def linear_alpha_schedule(timesteps: int) -> torch.Tensor:
    """Blur accumulation schedule: alpha linearly grows from 1 to 2."""
    return torch.linspace(1.0, 2.0, timesteps + 1)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class LatentExposureDiffusion(nn.Module):
    """Dual noise-and-blur latent diffusion prior for image deblurring.

    The forward (noising) process adds both Gaussian noise and progressive
    blur to a sharp latent code.  The reverse process jointly denoises and
    deblurs using two dedicated MLP heads.

    Args:
        total_timestamps:    Number of diffusion steps T (default 5).
        supervised_mid_out:  Return intermediate predictions for auxiliary loss.
        n_feats:             Base encoder feature width (default 64).
        n_res_blocks:        ResMLP depth per estimator head (default 5).
        time_emb_dim:        Sinusoidal embedding dimension (default 64).
        noise_schedule:      'cosine' (recommended) or 'linear'.
    """

    def __init__(
        self,
        total_timestamps: int = 5,
        supervised_mid_out: bool = False,
        n_feats: int = 64,
        n_res_blocks: int = 5,
        time_emb_dim: int = 64,
        noise_schedule: str = "cosine",
    ) -> None:
        super().__init__()

        self.total_timestamps = total_timestamps
        self.supervised_mid_out = supervised_mid_out

        # Dual estimator heads
        self.blur_residual_model = DenoiseHead(n_feats, n_res_blocks, time_emb_dim)
        self.noise_model         = DenoiseHead(n_feats, n_res_blocks, time_emb_dim)
        self.condition_encoder   = LE_arch()

        # Build and register noise schedule buffers
        betas = (
            cosine_beta_schedule(total_timestamps)
            if noise_schedule == "cosine"
            else linear_beta_schedule(total_timestamps)
        )
        alphas   = linear_alpha_schedule(total_timestamps)
        betas_bar = self._compute_betas_bar(alphas, betas)

        self.register_buffer("alphas",    alphas)
        self.register_buffer("betas",     betas)
        self.register_buffer("betas_bar", betas_bar)
        self.register_buffer(
            "timestep_seq",
            torch.arange(total_timestamps, 0, -1, dtype=torch.long),
        )

    # ------------------------------------------------------------------
    # Schedule construction
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_betas_bar(
        alphas: torch.Tensor, betas: torch.Tensor
    ) -> torch.Tensor:
        """Cumulative blur-weighted noise std at each step t."""
        T = len(betas)
        out = torch.zeros(T)
        for t in range(1, T + 1):
            w = torch.tensor(
                [(float(alphas[i - 1]) / float(alphas[t])) ** 2 for i in range(1, t + 1)]
            )
            out[t - 1] = (w * betas[:t]).sum().sqrt()
        return out

    # ------------------------------------------------------------------
    # Forward (noising) process
    # ------------------------------------------------------------------

    def q_sample(
        self,
        latent: torch.Tensor,
        t_idx: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a noisy latent at diffusion step *t_idx*.

        Args:
            latent: (B, C) clean latent.
            t_idx:  Step index in [0, T-1]. Defaults to T-1 (maximum noise).
        Returns:
            Tuple of (noisy_latent, noise), each shape (B, C).
        """
        idx = (self.total_timestamps - 1) if t_idx is None else t_idx
        noise = torch.randn_like(latent)
        return latent + self.betas_bar[idx] * noise, noise

    # ------------------------------------------------------------------
    # Single reverse step
    # ------------------------------------------------------------------

    def _reverse_step(
        self,
        z: torch.Tensor,
        i: int,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One reverse denoising+deblurring step.

        Args:
            z:         (B, C) current noisy latent.
            i:         Integer 1-indexed timestep (counts down to 1).
            condition: (B, C) encoded condition.
        Returns:
            Tuple of (updated_z, pred_lcr, pred_noise).
        """
        B, device = z.shape[0], z.device
        t = torch.full((B,), i, device=device, dtype=torch.long)

        pred_noise = self.noise_model(z, t, condition)
        pred_lcr   = self.blur_residual_model(z, t, condition)

        alpha_t   = self.alphas[i]
        alpha_tm1 = self.alphas[i - 1]
        beta_bar  = self.betas_bar[i - 1]

        if i == 1:
            noise_coef = beta_bar
        else:
            noise_coef = (alpha_t * beta_bar) / alpha_tm1 - self.betas_bar[i - 2]

        z_next = (alpha_t * z - pred_lcr) / alpha_tm1 - noise_coef * pred_noise
        return z_next, pred_lcr, pred_noise

    # ------------------------------------------------------------------
    # Full forward pass
    # ------------------------------------------------------------------

    def forward(self, blur: torch.Tensor):
        """Generate a sharp latent prior from a blurry input.

        Args:
            blur: (B, 3, H, W) blurry image in [-0.5, 0.5].
        Returns:
            If supervised_mid_out=False: (B, C) sharp latent prior.
            If supervised_mid_out=True:  (sharp_latent, lcr_list, noise_list, gt_noise).
        """
        condition = self.condition_encoder(blur, blur)   # (B, C)
        z, gt_noise = self.q_sample(condition)           # start from full noise

        lcr_list: list[torch.Tensor] = []
        noise_list: list[torch.Tensor] = []

        for i in self.timestep_seq:                      # T, T-1, ..., 1
            z, pred_lcr, pred_noise = self._reverse_step(z, int(i.item()), condition)
            if self.supervised_mid_out:
                lcr_list.append(pred_lcr)
                noise_list.append(pred_noise)

        if self.supervised_mid_out:
            return z, lcr_list, noise_list, gt_noise
        return z
