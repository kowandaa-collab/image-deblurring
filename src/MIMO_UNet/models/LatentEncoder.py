"""
LatentEncoder (LE_arch): encodes a blurry/sharp image pair into a condition vector.

Improvements over the original:
  - Removed hardcoded sys.path.append('/home/jthe/FFTformer')
  - Type annotations and docstrings
  - Configurable n_feats and n_encoder_res
  - Uses nn.GELU instead of LeakyReLU in the MLP (modern default)
  - PixelUnshuffle downscale factor exposed as argument
"""

import torch
import torch.nn as nn


def _default_conv(in_ch: int, out_ch: int, kernel_size: int, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=bias)


class ResBlock(nn.Module):
    """Residual conv block used inside the encoder."""

    def __init__(
        self,
        n_feats: int,
        kernel_size: int = 3,
        act: nn.Module | None = None,
    ) -> None:
        super().__init__()
        act = act or nn.LeakyReLU(0.1, inplace=True)
        self.body = nn.Sequential(
            _default_conv(n_feats, n_feats, kernel_size),
            act,
            _default_conv(n_feats, n_feats, kernel_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class LE_arch(nn.Module):
    """Latent Encoder: maps a (blur, reference) pair to a global condition vector.

    The two images are pixel-unshuffled and concatenated before encoding,
    allowing the network to see both the degraded input and a reference.

    Args:
        n_feats:        Base feature width.
        n_encoder_res:  Number of residual blocks in the trunk.
        unshuffle_factor: Spatial downscaling factor for PixelUnshuffle.
    """

    def __init__(
        self,
        n_feats: int = 64,
        n_encoder_res: int = 6,
        unshuffle_factor: int = 4,
    ) -> None:
        super().__init__()

        self.pixel_unshuffle = nn.PixelUnshuffle(unshuffle_factor)

        # After unshuffle, each image has 3 * unshuffle_factor^2 channels.
        in_ch = 3 * (unshuffle_factor ** 2) * 2  # two images concatenated

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, n_feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Residual trunk
        self.trunk = nn.Sequential(
            *[ResBlock(n_feats) for _ in range(n_encoder_res)]
        )

        # Projection head: spatial → global vector
        self.head = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * 2, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(n_feats * 2, n_feats * 2, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(n_feats * 2, n_feats * 4, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        # MLP on the pooled vector (GELU for smoother gradients)
        self.mlp = nn.Sequential(
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.GELU(),
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Encode (blurry, reference) pair.

        Args:
            x:  (B, 3, H, W) blurry image.
            gt: (B, 3, H, W) reference image (can equal x at inference).
        Returns:
            (B, n_feats*4) condition vector.
        """
        x_us  = self.pixel_unshuffle(x)
        gt_us = self.pixel_unshuffle(gt)
        feat  = torch.cat([x_us, gt_us], dim=1)
        feat  = self.stem(feat)
        feat  = self.trunk(feat)
        feat  = self.head(feat).flatten(1)  # (B, n_feats*4)
        return self.mlp(feat)
