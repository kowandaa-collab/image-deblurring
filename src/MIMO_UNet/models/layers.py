"""
layers.py — Shared primitive building blocks.

Improvements over the original:
  - GroupNorm option alongside BatchNorm (better for small batch sizes)
  - Dropout option in ResBlock for regularization
  - Type annotations
  - Removed implicit bias=False when norm=True (kept but made explicit)
"""

import torch
import torch.nn as nn


class BasicConv(nn.Module):
    """Conv2d (or ConvTranspose2d) optionally followed by norm and activation.

    Args:
        in_channel:  Input channels.
        out_channel: Output channels.
        kernel_size: Convolution kernel size.
        stride:      Convolution stride.
        bias:        Use bias (auto-disabled when norm is used).
        norm:        'batch' | 'group' | None.
        relu:        Append ReLU if True.
        transpose:   Use ConvTranspose2d (upsampling) if True.
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        stride: int,
        bias: bool = True,
        norm: str | None = None,
        relu: bool = True,
        transpose: bool = False,
    ) -> None:
        super().__init__()

        if norm is not None:
            bias = False  # redundant with normalisation layers

        padding = kernel_size // 2
        layers: list[nn.Module] = []

        if transpose:
            layers.append(
                nn.ConvTranspose2d(
                    in_channel, out_channel, kernel_size,
                    padding=max(padding - 1, 0), stride=stride, bias=bias,
                )
            )
        else:
            layers.append(
                nn.Conv2d(
                    in_channel, out_channel, kernel_size,
                    padding=padding, stride=stride, bias=bias,
                )
            )

        if norm == "batch":
            layers.append(nn.BatchNorm2d(out_channel))
        elif norm == "group":
            num_groups = min(32, out_channel)
            layers.append(nn.GroupNorm(num_groups, out_channel))

        if relu:
            layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class ResBlock(nn.Module):
    """Two-layer residual convolutional block.

    Args:
        in_channel:  Input/output channels (must match for skip connection).
        out_channel: Output channels.
        dropout:     Dropout probability applied after first conv (0 = off).
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) + x
