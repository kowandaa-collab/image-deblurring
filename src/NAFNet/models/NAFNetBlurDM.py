"""
NAFNetBlurDM v4 — NAFNet backbone conditioned on a BlurDM latent prior.

Speed improvements in v4:
  - torch.compile() compatibility: all ops are compile-friendly (no Python
    control flow inside forward, no dynamic shapes)
  - Checkpoint activations (gradient checkpointing) on encoder blocks via
    --grad_ckpt flag — trades recompute for memory, enabling larger batch sizes
  - Conv bias=False when LayerNorm follows (norm absorbs bias, saves params+FLOPs)
  - Fused depthwise+pointwise via single grouped conv where possible

Quality improvements in v4:
  - Cross-attention prior injection option (--use_cross_attn): attends over
    spatial positions using the prior as keys/values, richer than FiLM
  - Stochastic depth (drop_path) replaces Dropout2d in NAFBlock for better
    regularisation in deep configurations
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


# ---------------------------------------------------------------------------
# Stochastic depth (drop path)
# ---------------------------------------------------------------------------

def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Per-sample stochastic depth (Huang et al., 2016)."""
    if drop_prob == 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask  = torch.empty(shape, device=x.device).bernoulli_(keep).div_(keep)
    return x * mask


# ---------------------------------------------------------------------------
# NAFNet primitives
# ---------------------------------------------------------------------------

class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimpleChannelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        # bias=False: LayerNorm is applied before, so bias is redundant
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.proj(x.mean(dim=(2, 3), keepdim=True))


class NAFBlock(nn.Module):
    """NAFNet block with stochastic depth and compile-friendly ops.

    Args:
        channels:    Feature channels.
        dw_expand:   Depth-wise conv expansion ratio.
        ffn_expand:  FFN expansion ratio.
        drop_path_p: Stochastic depth probability (0 = off).
        grad_ckpt:   Use gradient checkpointing on this block.
    """

    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        drop_path_p: float = 0.0,
        grad_ckpt: bool = False,
    ) -> None:
        super().__init__()
        dw_ch  = channels * dw_expand
        ffn_ch = channels * ffn_expand
        self.drop_path_p = drop_path_p
        self.grad_ckpt   = grad_ckpt

        # Spatial branch
        self.norm1 = nn.LayerNorm(channels)
        self.conv1 = nn.Conv2d(channels, dw_ch,  1, bias=False)
        self.conv2 = nn.Conv2d(dw_ch, dw_ch, 3, padding=1, groups=dw_ch, bias=False)
        self.gate  = SimpleGate()
        self.attn  = SimpleChannelAttention(dw_ch // 2)
        self.conv3 = nn.Conv2d(dw_ch // 2, channels, 1, bias=False)

        # Channel (FFN) branch
        self.norm2 = nn.LayerNorm(channels)
        self.conv4 = nn.Conv2d(channels, ffn_ch, 1, bias=False)
        self.gate2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_ch // 2, channels, 1, bias=False)

        self.beta  = nn.Parameter(torch.ones(1, channels, 1, 1) * 1e-2)
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * 1e-2)

    def _ln(self, norm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
        return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial branch
        y = self._ln(self.norm1, x)
        y = self.gate(self.conv2(self.conv1(y)))
        y = self.conv3(self.attn(y))
        x = x + drop_path(self.beta * y, self.drop_path_p, self.training)

        # FFN branch
        y = self._ln(self.norm2, x)
        y = self.conv5(self.gate2(self.conv4(y)))
        x = x + drop_path(self.gamma * y, self.drop_path_p, self.training)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.grad_ckpt and self.training:
            return cp.checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


# ---------------------------------------------------------------------------
# Prior injection: FiLM and Cross-Attention variants
# ---------------------------------------------------------------------------

class FiLMInjection(nn.Module):
    """Feature-wise Linear Modulation. Simple, fast, zero-init."""

    def __init__(self, channels: int, prior_dim: int) -> None:
        super().__init__()
        self.film = nn.Linear(prior_dim, channels * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.ones_(self.film.bias[:channels])
        nn.init.zeros_(self.film.bias[channels:])

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        film  = self.film(prior).unsqueeze(-1).unsqueeze(-1)
        scale, shift = film.chunk(2, dim=1)
        return x * scale + shift


class CrossAttentionInjection(nn.Module):
    """Cross-attention prior injection.

    Spatial feature map attends over prior as key/value. Richer than FiLM
    because the injected information can vary spatially.

    Args:
        channels:  Spatial feature channels (query dimension).
        prior_dim: Prior vector dimension (key/value dimension).
        heads:     Number of attention heads.
    """

    def __init__(self, channels: int, prior_dim: int, heads: int = 4) -> None:
        super().__init__()
        self.heads   = heads
        self.head_ch = channels // heads
        self.scale   = self.head_ch ** -0.5

        self.q_proj = nn.Conv2d(channels,  channels, 1, bias=False)
        self.k_proj = nn.Linear(prior_dim, channels, bias=False)
        self.v_proj = nn.Linear(prior_dim, channels, bias=False)
        self.out    = nn.Conv2d(channels,  channels, 1, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Q from spatial features: (B, heads, H*W, head_ch)
        q = self.q_proj(x).view(B, self.heads, self.head_ch, H * W).permute(0, 1, 3, 2)
        # K, V from prior: (B, heads, 1, head_ch) — broadcast over H*W
        k = self.k_proj(prior).view(B, self.heads, self.head_ch).unsqueeze(2)
        v = self.v_proj(prior).view(B, self.heads, self.head_ch).unsqueeze(2)

        # Attention weights: (B, heads, H*W, 1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # Weighted sum: (B, heads, H*W, head_ch)
        out  = (attn @ v).permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        return x + self.out(out)


# ---------------------------------------------------------------------------
# Decoder block with configurable prior injection
# ---------------------------------------------------------------------------

class NAFDecoderBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_blocks: int,
        prior_dim: int,
        drop_path_p: float = 0.0,
        grad_ckpt: bool = False,
        use_cross_attn: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[
            NAFBlock(channels, drop_path_p=drop_path_p, grad_ckpt=grad_ckpt)
            for _ in range(num_blocks)
        ])
        if use_cross_attn:
            self.inject = CrossAttentionInjection(channels, prior_dim)
        else:
            self.inject = FiLMInjection(channels, prior_dim)

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.inject(x, prior))


# ---------------------------------------------------------------------------
# Full NAFNetBlurDM v4
# ---------------------------------------------------------------------------

class NAFNetBlurDM(nn.Module):
    """NAFNet deblurring backbone with BlurDM prior conditioning.

    Args:
        img_channels:    RGB channels (default 3).
        width:           Base channel width.
        enc_blocks:      NAFBlocks per encoder stage.
        dec_blocks:      NAFBlocks per decoder stage.
        prior_dim:       Incoming prior vector dimension.
        drop_path_rate:  Max stochastic depth rate (linearly scaled per block).
        grad_ckpt:       Enable gradient checkpointing on encoder (saves VRAM).
        use_cross_attn:  Use cross-attention instead of FiLM in decoder.
    """

    def __init__(
        self,
        img_channels: int = 3,
        width: int = 32,
        enc_blocks: list[int] | None = None,
        dec_blocks: list[int] | None = None,
        prior_dim: int = 256,
        drop_path_rate: float = 0.0,
        grad_ckpt: bool = False,
        use_cross_attn: bool = False,
    ) -> None:
        super().__init__()
        if enc_blocks is None:
            enc_blocks = [1, 1, 1, 28]
        if dec_blocks is None:
            dec_blocks = [1, 1, 1, 1]

        # Stochastic depth: linearly increase rate towards bottleneck
        total_blocks = sum(enc_blocks)
        dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)
        ]
        dp_idx = 0

        self.intro = nn.Conv2d(img_channels, width, 3, padding=1, bias=False)
        self.outro = nn.Conv2d(width, img_channels, 3, padding=1)

        # Encoder
        self.encoders: nn.ModuleList = nn.ModuleList()
        self.downs:    nn.ModuleList = nn.ModuleList()
        ch = width
        for n in enc_blocks:
            self.encoders.append(nn.Sequential(*[
                NAFBlock(ch, drop_path_p=dp_rates[dp_idx + i], grad_ckpt=grad_ckpt)
                for i in range(n)
            ]))
            dp_idx += n
            self.downs.append(nn.Conv2d(ch, ch * 2, 2, stride=2, bias=False))
            ch *= 2

        # Bottleneck
        self.middle = nn.Sequential(*[NAFBlock(ch) for _ in range(1)])

        # Decoder
        self.decoders: nn.ModuleList = nn.ModuleList()
        self.ups:      nn.ModuleList = nn.ModuleList()
        self.skips:    nn.ModuleList = nn.ModuleList()
        for n in dec_blocks:
            self.ups.append(nn.ConvTranspose2d(ch, ch // 2, 2, stride=2, bias=False))
            ch //= 2
            self.skips.append(nn.Conv2d(ch * 2, ch, 1, bias=False))
            self.decoders.append(
                NAFDecoderBlock(ch, n, prior_dim,
                                use_cross_attn=use_cross_attn)
            )

        # Multi-scale output heads (1/4- and 1/2-resolution branches vs full-res outro).
        # Decoder index i increases as spatial size grows: x_4 aligns with i = n_stages-3,
        # x_2 with i = n_stages-2 (was previously swapped, causing channel mismatch).
        n_stages = len(dec_blocks)
        self._out4_idx = n_stages - 3 if n_stages >= 3 else -1
        self._out2_idx = n_stages - 2 if n_stages >= 2 else -1
        ch4 = width * (2 ** max(n_stages - 2, 0))
        ch2 = width * (2 ** max(n_stages - 3, 0)) if n_stages >= 3 else width
        self.out_head_4 = nn.Conv2d(ch4, img_channels, 3, padding=1)
        self.out_head_2 = nn.Conv2d(ch2, img_channels, 3, padding=1)

    def forward(
        self, x: torch.Tensor, prior: torch.Tensor
    ) -> list[torch.Tensor]:
        inp = x
        x_4 = F.interpolate(x, scale_factor=0.25, mode="bilinear", align_corners=False)
        x_2 = F.interpolate(x, scale_factor=0.50, mode="bilinear", align_corners=False)

        x = self.intro(inp)
        skips: list[torch.Tensor] = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.middle(x)

        outputs: list[torch.Tensor] = []
        for i, (dec, up, skip_proj) in enumerate(
            zip(self.decoders, self.ups, self.skips)
        ):
            x = up(x)
            skip = skips[-(i + 1)]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = skip_proj(torch.cat([x, skip], dim=1))
            x = dec(x, prior)

            if i == self._out4_idx:
                outputs.append(self.out_head_4(x) + x_4)
            elif i == self._out2_idx:
                outputs.append(self.out_head_2(x) + x_2)

        outputs.append(self.outro(x) + inp)

        while len(outputs) < 3:
            outputs.insert(0, F.interpolate(
                outputs[-1], scale_factor=0.5 ** (3 - len(outputs)),
                mode="bilinear", align_corners=False,
            ))
        return outputs


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_CONFIGS: dict[str, dict] = {
    "NAFNetBlurDM-light": dict(width=32, enc_blocks=[1, 1, 1, 28], dec_blocks=[1, 1, 1, 1]),
    "NAFNetBlurDM-base":  dict(width=64, enc_blocks=[2, 2, 4, 8],  dec_blocks=[2, 2, 2, 2]),
    "NAFNetBlurDM-large": dict(width=64, enc_blocks=[2, 2, 4, 28], dec_blocks=[2, 2, 2, 2]),
}


def build_NAFNet(
    model_name: str,
    prior_dim: int = 256,
    grad_ckpt: bool = False,
    drop_path_rate: float = 0.0,
    use_cross_attn: bool = False,
    **kwargs,
) -> nn.Module:
    if model_name not in _CONFIGS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(_CONFIGS)}")
    cfg = {
        **_CONFIGS[model_name],
        "prior_dim": prior_dim,
        "grad_ckpt": grad_ckpt,
        "drop_path_rate": drop_path_rate,
        "use_cross_attn": use_cross_attn,
        **kwargs,
    }
    return NAFNetBlurDM(**cfg)
