"""
LatentEncoder (LE_arch): encodes a blurry/sharp image pair into a condition vector.

Multi-scale residual encoder with layer-scale blocks and attention pooling.
This layout matches the checkpoints produced by BlurDM stage-1/2/3 training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _default_conv(
    in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, bias: bool = True
) -> nn.Conv2d:
    return nn.Conv2d(
        in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2, bias=bias
    )


class ScaleResBlock(nn.Module):
    """Residual conv block with per-channel layer scale (gamma)."""

    def __init__(self, n_feats: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n_feats))
        self.body = nn.Sequential(
            _default_conv(n_feats, n_feats),
            nn.LeakyReLU(0.1, inplace=True),
            _default_conv(n_feats, n_feats),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma.view(1, -1, 1, 1) * self.body(x)


class SpatialAttentionPool(nn.Module):
    """Softmax-weighted global pooling over spatial locations."""

    def __init__(self, in_ch: int = 256, mid_ch: int = 64) -> None:
        super().__init__()
        self.score = nn.Sequential(
            _default_conv(in_ch, mid_ch, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            _default_conv(mid_ch, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        attn = F.softmax(self.score(x).view(b, -1), dim=1)
        feat = x.view(b, c, -1)
        return (feat * attn.unsqueeze(1)).sum(dim=2)


class LE_arch(nn.Module):
    """Latent Encoder: maps a (blur, reference) pair to a global condition vector.

    Args:
        n_feats:        Base feature width (64).
        n_blocks:       Residual blocks per scale (3).
        unshuffle_factor: PixelUnshuffle factor (4 → 96 input channels).
    """

    def __init__(
        self,
        n_feats: int = 64,
        n_blocks: int = 3,
        unshuffle_factor: int = 4,
    ) -> None:
        super().__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(unshuffle_factor)
        in_ch = 3 * (unshuffle_factor ** 2) * 2

        self.stem = nn.Sequential(
            _default_conv(in_ch, n_feats),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.scale0 = nn.Sequential(*[ScaleResBlock(n_feats) for _ in range(n_blocks)])
        self.down0 = _default_conv(n_feats, n_feats * 2, stride=2)
        self.scale1 = nn.Sequential(
            *[ScaleResBlock(n_feats * 2) for _ in range(n_blocks)]
        )
        self.down1 = _default_conv(n_feats * 2, n_feats * 4, stride=2)
        self.scale2 = nn.Sequential(
            *[ScaleResBlock(n_feats * 4) for _ in range(n_blocks)]
        )
        self.proj0 = _default_conv(n_feats, n_feats * 4, kernel_size=1)
        self.proj1 = _default_conv(n_feats * 2, n_feats * 4, kernel_size=1)
        self.pool = SpatialAttentionPool(in_ch=n_feats * 4)
        # Indices 0 and 3 are Linear layers (matches saved checkpoints).
        self.mlp = nn.Sequential(
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.GELU(),
            nn.Identity(),
            nn.Linear(n_feats * 4, n_feats * 4),
        )

    def forward(self, x: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        x_us = self.pixel_unshuffle(x)
        gt_us = self.pixel_unshuffle(gt)
        feat = torch.cat([x_us, gt_us], dim=1)

        f0 = self.scale0(self.stem(feat))
        f1 = self.scale1(self.down0(f0))
        f2 = self.scale2(self.down1(f1))

        h, w = f2.shape[-2:]
        fused = (
            f2
            + F.interpolate(self.proj0(f0), size=(h, w), mode="bilinear", align_corners=False)
            + F.interpolate(self.proj1(f1), size=(h, w), mode="bilinear", align_corners=False)
        )
        return self.mlp(self.pool(fused))
