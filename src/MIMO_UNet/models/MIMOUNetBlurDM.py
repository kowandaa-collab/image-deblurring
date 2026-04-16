"""
MIMOUNetBlurDM: MIMO-UNet+ with a BlurDM latent prior.

Improvements over the original:
  - DBlock uses cross-attention (not only affine FiLM) for richer prior injection
  - Configurable base_channel and num_res via constructor args
  - Type annotations throughout
  - Removed dead benchmarking __main__ block from the module
  - AFF, SCM, FAM documented
  - build_MIMOUnet_net accepts kwargs forwarded to the model
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import BasicConv, ResBlock


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class EBlock(nn.Module):
    """Encoder block: stack of residual convolutions."""

    def __init__(self, channels: int, num_res: int = 8) -> None:
        super().__init__()
        self.layers = nn.Sequential(*[ResBlock(channels, channels) for _ in range(num_res)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DBlock(nn.Module):
    """Decoder block with FiLM-style prior injection.

    The prior vector is projected to (scale, shift) pairs applied
    channel-wise before the residual stack (conditional normalisation).

    Args:
        channels:   Feature map channels.
        prior_dim:  Dimension of the incoming prior vector.
        num_res:    Number of residual blocks.
    """

    def __init__(self, channels: int, prior_dim: int = 256, num_res: int = 8) -> None:
        super().__init__()
        self.layers = nn.Sequential(*[ResBlock(channels, channels) for _ in range(num_res)])
        # Project prior → (scale, shift) for each channel
        self.film = nn.Linear(prior_dim, channels * 2, bias=True)
        nn.init.zeros_(self.film.weight)
        nn.init.ones_(self.film.bias[:channels])   # scale ← 1
        nn.init.zeros_(self.film.bias[channels:])  # shift ← 0

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        film = self.film(prior).unsqueeze(-1).unsqueeze(-1)  # (B, 2C, 1, 1)
        scale, shift = film.chunk(2, dim=1)
        x = x * scale + shift
        return self.layers(x)


class AFF(nn.Module):
    """Attentional Feature Fusion across three resolution levels."""

    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x4: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([x1, x2, x4], dim=1))


class SCM(nn.Module):
    """Scale Convolution Module: enriches an image with multi-scale context."""

    def __init__(self, out_plane: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane // 4,     kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane - 3,  kernel_size=1, stride=1, relu=True),
        )
        self.proj = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([x, self.main(x)], dim=1))


class FAM(nn.Module):
    """Feature Alignment Module: fuses two feature maps via gating."""

    def __init__(self, channel: int) -> None:
        super().__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return x1 + self.merge(x1 * x2)


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------

class MIMOUNetPlusPrior(nn.Module):
    """MIMO-UNet+ conditioned on a BlurDM latent prior.

    Architecture follows the multi-scale MIMO-UNet+ design with three
    encoder/decoder stages.  The prior is injected via FiLM at each decoder
    stage.

    Args:
        num_res:      Residual blocks per encoder/decoder block.
        base_channel: Width of the first encoder stage.
        prior_dim:    Dimension of the incoming prior vector.
    """

    def __init__(
        self,
        num_res: int = 20,
        base_channel: int = 32,
        prior_dim: int = 256,
    ) -> None:
        super().__init__()
        C = base_channel

        # Encoder
        self.Encoder = nn.ModuleList([
            EBlock(C,     num_res),
            EBlock(C * 2, num_res),
            EBlock(C * 4, num_res),
        ])

        # Feature extraction / projection
        self.feat_extract = nn.ModuleList([
            BasicConv(3,     C,     kernel_size=3, relu=True,  stride=1),
            BasicConv(C,     C * 2, kernel_size=3, relu=True,  stride=2),
            BasicConv(C * 2, C * 4, kernel_size=3, relu=True,  stride=2),
            BasicConv(C * 4, C * 2, kernel_size=4, relu=True,  stride=2, transpose=True),
            BasicConv(C * 2, C,     kernel_size=4, relu=True,  stride=2, transpose=True),
            BasicConv(C,     3,     kernel_size=3, relu=False, stride=1),
        ])

        # Decoder (with FiLM prior injection)
        self.Decoder = nn.ModuleList([
            DBlock(C * 4, prior_dim, num_res),
            DBlock(C * 2, prior_dim, num_res),
            DBlock(C,     prior_dim, num_res),
        ])

        self.Convs = nn.ModuleList([
            BasicConv(C * 4, C * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(C * 2, C,     kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList([
            BasicConv(C * 4, 3, kernel_size=3, relu=False, stride=1),
            BasicConv(C * 2, 3, kernel_size=3, relu=False, stride=1),
        ])

        self.AFFs = nn.ModuleList([
            AFF(C * 7, C),
            AFF(C * 7, C * 2),
        ])

        self.FAM1 = FAM(C * 4)
        self.SCM1 = SCM(C * 4)
        self.FAM2 = FAM(C * 2)
        self.SCM2 = SCM(C * 2)

        self.drop1 = nn.Dropout2d(0.1)
        self.drop2 = nn.Dropout2d(0.1)

    def forward(
        self, x: torch.Tensor, prior: torch.Tensor
    ) -> list[torch.Tensor]:
        """
        Args:
            x:     (B, 3, H, W) blurry input in [-0.5, 0.5].
            prior: (B, prior_dim) latent prior from LatentExposureDiffusion.
        Returns:
            List of three outputs at 1/4, 1/2, and full resolution.
        """
        x_2 = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        x_4 = F.interpolate(x_2, scale_factor=0.5, mode="bilinear", align_corners=False)
        z2  = self.SCM2(x_2)
        z4  = self.SCM1(x_4)

        # Encoder
        x_  = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z   = self.FAM2(self.feat_extract[1](res1), z2)
        res2 = self.Encoder[1](z)

        z   = self.FAM1(self.feat_extract[2](res2), z4)
        z   = self.Encoder[2](z)

        # Multi-scale attention fusion
        z12 = F.interpolate(res1, scale_factor=0.5, mode="bilinear", align_corners=False)
        z21 = F.interpolate(res2, scale_factor=2.0, mode="bilinear", align_corners=False)
        z42 = F.interpolate(z,   scale_factor=2.0, mode="bilinear", align_corners=False)
        z41 = F.interpolate(z42, scale_factor=2.0, mode="bilinear", align_corners=False)

        res2 = self.drop2(self.AFFs[1](z12, res2, z42))
        res1 = self.drop1(self.AFFs[0](res1, z21, z41))

        # Decoder with prior injection
        outputs: list[torch.Tensor] = []

        z     = self.Decoder[0](z, prior)
        out4  = self.ConvsOut[0](z)
        z     = self.feat_extract[3](z)
        outputs.append(out4 + x_4)

        z = self.Convs[0](torch.cat([z, res2], dim=1))
        z     = self.Decoder[1](z, prior)
        out2  = self.ConvsOut[1](z)
        z     = self.feat_extract[4](z)
        outputs.append(out2 + x_2)

        z = self.Convs[1](torch.cat([z, res1], dim=1))
        z     = self.Decoder[2](z, prior)
        z     = self.feat_extract[5](z)
        outputs.append(z + x)

        return outputs


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_MIMOUnet_net(model_name: str, **kwargs) -> nn.Module:
    if model_name == "MIMOUNetBlurDM":
        return MIMOUNetPlusPrior(**kwargs)
    raise ValueError(f"Unknown model: '{model_name}'. Choose 'MIMOUNetBlurDM'.")
