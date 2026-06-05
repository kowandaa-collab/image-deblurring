"""
losses.py — Loss functions for BlurDM training.

Improvements over the original:
  - VGG loaded with weights='DEFAULT' (deprecation fix for pretrained=True)
  - PerceptualLoss unified into one class (removed the tab-indented duplicate)
  - Type annotations
  - PSNRLoss sign made explicit in docstring (it is a *loss*, so lower = better)
  - GANLoss handles non-hinge modes cleanly
  - All classes use nn.Module consistently
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Pixel-space losses
# ---------------------------------------------------------------------------

class CharbonnierLoss(nn.Module):
    """Charbonnier (pseudo-L1) loss: sqrt((x-y)^2 + eps^2).

    Smoother than L1 near zero; more robust to outliers than L2.
    """

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class PSNRLoss(nn.Module):
    """Negative log-MSE loss that approximates -PSNR (lower = sharper output).

    Args:
        loss_weight: Scalar multiplier.
        toY:         Convert to Y-channel before computing (for PSNR on luma).
    """

    def __init__(self, loss_weight: float = 1.0, toY: bool = False) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.scale = 10.0 / np.log(10)
        self.toY = toY
        self.register_buffer(
            "coef", torch.tensor([65.481, 128.553, 24.966]).view(1, 3, 1, 1)
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.toY:
            pred   = (pred   * self.coef).sum(1, keepdim=True) + 16.0
            target = (target * self.coef).sum(1, keepdim=True) + 16.0
            pred, target = pred / 255.0, target / 255.0
        mse = ((pred - target) ** 2).mean(dim=(1, 2, 3))
        return self.loss_weight * self.scale * torch.log(mse + 1e-8).mean()


# ---------------------------------------------------------------------------
# Perceptual losses
# ---------------------------------------------------------------------------

class VGGPerceptualLoss(nn.Module):
    """Multi-layer VGG16 feature-matching loss.

    Args:
        feature_layers: VGG block indices to include in the loss (default all 4).
        style_layers:   Block indices for optional Gram-matrix style loss.
        resize:         Resize inputs to 224×224 before passing through VGG.
    """

    def __init__(
        self,
        feature_layers: list[int] | None = None,
        style_layers: list[int] | None = None,
        resize: bool = False,
    ) -> None:
        super().__init__()
        if feature_layers is None:
            feature_layers = [0, 1, 2, 3]
        if style_layers is None:
            style_layers = []

        self.feature_layers = feature_layers
        self.style_layers   = style_layers
        self.resize         = resize

        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
        slices = [
            vgg.features[:4].eval(),
            vgg.features[4:9].eval(),
            vgg.features[9:16].eval(),
            vgg.features[16:23].eval(),
        ]
        for s in slices:
            for p in s.parameters():
                p.requires_grad_(False)
        self.blocks = nn.ModuleList(slices)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if pred.shape[1] != 3:
            pred   = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        pred   = (pred   - self.mean) / self.std
        target = (target - self.mean) / self.std

        if self.resize:
            pred   = F.interpolate(pred,   size=(224, 224), mode="bilinear", align_corners=False)
            target = F.interpolate(target, size=(224, 224), mode="bilinear", align_corners=False)

        loss = pred.new_zeros(1)
        x, y = pred, target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in self.feature_layers:
                loss = loss + F.l1_loss(x, y)
            if i in self.style_layers:
                gx = x.flatten(2) @ x.flatten(2).transpose(1, 2)
                gy = y.flatten(2) @ y.flatten(2).transpose(1, 2)
                loss = loss + F.l1_loss(gx, gy)
        return loss


class L1andPerceptualLoss(nn.Module):
    """Charbonnier + weighted VGG perceptual loss.

    Args:
        gamma: Weight on the perceptual term (default 0.1).
    """

    def __init__(self, gamma: float = 0.1) -> None:
        super().__init__()
        self.l1          = CharbonnierLoss()
        self.perceptual  = VGGPerceptualLoss()
        self.gamma       = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.l1(pred, target) + self.gamma * self.perceptual(pred, target)


# ---------------------------------------------------------------------------
# Adversarial losses
# ---------------------------------------------------------------------------

class GANLoss(nn.Module):
    """Hinge or vanilla GAN loss.

    Args:
        gan_mode: 'hinge' or 'vanilla'.
    """

    def __init__(self, gan_mode: str = "hinge") -> None:
        super().__init__()
        if gan_mode not in ("hinge", "vanilla"):
            raise ValueError(f"Unsupported gan_mode: {gan_mode}")
        self.gan_mode = gan_mode

    def forward(
        self,
        logits: torch.Tensor,
        target_is_real: bool,
        for_discriminator: bool = False,
    ) -> torch.Tensor:
        if self.gan_mode == "hinge":
            if for_discriminator:
                sign = 1.0 if target_is_real else -1.0
                return -torch.mean(torch.clamp(sign * logits - 1.0, max=0.0))
            # generator always wants to fool discriminator
            return -logits.mean()

        # vanilla BCE-style
        labels = torch.ones_like(logits) if target_is_real else torch.zeros_like(logits)
        return F.binary_cross_entropy_with_logits(logits, labels)


class DeblurLoss(nn.Module):
    """Combined deblurring loss: Charbonnier + optional edge/SSIM/focal-freq/perceptual.

    Uses learnable uncertainty weighting (log_vars) to auto-balance terms.
    """

    def __init__(
        self,
        use_perceptual: bool = False,
        use_edge: bool = True,
        use_ssim: bool = True,
        use_focal_freq: bool = True,
    ) -> None:
        super().__init__()
        self.charbonnier    = CharbonnierLoss()
        self.use_perceptual = use_perceptual
        self.use_edge       = use_edge
        self.use_ssim       = use_ssim
        self.use_focal_freq = use_focal_freq
        if use_perceptual:
            self.perceptual = VGGPerceptualLoss()
        n_terms = 1 + int(use_edge) + int(use_ssim) + int(use_focal_freq) + int(use_perceptual)
        self.log_vars = nn.Parameter(torch.zeros(n_terms))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses: list[torch.Tensor] = [self.charbonnier(pred, target)]
        if self.use_edge:
            losses.append(self._edge_loss(pred, target))
        if self.use_ssim:
            losses.append(self._ssim_loss(pred, target))
        if self.use_focal_freq:
            losses.append(self._focal_freq_loss(pred, target))
        if self.use_perceptual:
            losses.append(self.perceptual(pred, target))
        return sum(
            torch.exp(-self.log_vars[i]) * l + self.log_vars[i]
            for i, l in enumerate(losses)
        )

    @staticmethod
    def _edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        sy = sx.transpose(2, 3)
        def edges(x: torch.Tensor) -> torch.Tensor:
            luma = x.mean(1, keepdim=True)
            return torch.sqrt(F.conv2d(luma, sx, padding=1)**2
                              + F.conv2d(luma, sy, padding=1)**2 + 1e-6)
        return F.l1_loss(edges(pred), edges(target))

    @staticmethod
    def _ssim_loss(pred: torch.Tensor, target: torch.Tensor,
                   window_size: int = 11) -> torch.Tensor:
        C1, C2 = 0.01**2, 0.03**2
        coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device) - window_size // 2
        g = torch.exp(-(coords**2) / (2 * 1.5**2))
        g = g / g.sum()
        win = g.outer(g).unsqueeze(0).unsqueeze(0)
        C   = pred.shape[1]
        win = win.expand(C, 1, window_size, window_size)
        pad = window_size // 2
        mu1 = F.conv2d(pred,   win, padding=pad, groups=C)
        mu2 = F.conv2d(target, win, padding=pad, groups=C)
        s1  = F.conv2d(pred*pred,     win, padding=pad, groups=C) - mu1**2
        s2  = F.conv2d(target*target, win, padding=pad, groups=C) - mu2**2
        s12 = F.conv2d(pred*target,   win, padding=pad, groups=C) - mu1*mu2
        ssim = ((2*mu1*mu2 + C1) * (2*s12 + C2)) / ((mu1**2 + mu2**2 + C1) * (s1 + s2 + C2))
        return 1.0 - ssim.mean()

    @staticmethod
    def _focal_freq_loss(pred: torch.Tensor, target: torch.Tensor,
                         alpha: float = 1.0) -> torch.Tensor:
        pf   = torch.fft.rfft2(pred,   norm="ortho")
        tf   = torch.fft.rfft2(target, norm="ortho")
        diff = (pf - tf).abs()
        w    = (diff.detach() ** alpha)
        w    = w / (w.mean() + 1e-8)
        return (w * diff).mean()


def wgan_gp_loss(
    discriminator: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
) -> torch.Tensor:
    """Gradient penalty for WGAN-GP (Gulrajani et al., 2017)."""
    B = real.shape[0]
    alpha = torch.rand(B, 1, 1, 1, device=real.device).expand_as(real)
    interp = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    d_interp = discriminator(interp)
    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]
    return ((grads.flatten(1).norm(2, dim=1) - 1.0) ** 2).mean()
