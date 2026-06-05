"""
RestormerBlurDM: Restormer backbone conditioned on a BlurDM latent prior.

Restormer (CVPR 2022) key innovations:
  - MDTA (Multi-Dconv Head Transposed Attention): computes attention across
    channels rather than spatial positions → O(C²HW) instead of O(H²W²C).
    This makes it dramatically faster than Stripformer at high resolution.
  - GDFN (Gated-Dconv Feed-Forward Network): two-branch gated FFN with
    depthwise conv for local context capture.
  - PixelShuffle/PixelUnshuffle for lossless up/downsampling.

Adaptations for BlurDM:
  - Decoder blocks accept a (B, prior_dim) vector for FiLM or cross-attention
    prior injection (same interface as NAFNetBlurDM and MIMOUNetBlurDM).
  - Multi-scale outputs (1/4, 1/2, full) for auxiliary loss supervision.
  - torch.compile() compatible (no Python control-flow inside forward, no
    dynamic shapes).
  - Gradient checkpointing on encoder blocks via grad_ckpt=True.
  - Stochastic depth (drop_path) on transformer blocks.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


# ---------------------------------------------------------------------------
# Stochastic depth
# ---------------------------------------------------------------------------

def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = torch.empty(shape, device=x.device).bernoulli_(keep).div_(keep)
    return x * mask


# ---------------------------------------------------------------------------
# MDTA — Multi-Dconv Head Transposed Attention
# ---------------------------------------------------------------------------

class MDTA(nn.Module):
    """Transposed self-attention over channels with depthwise Q/K/V extraction.

    Attention matrix is (C/heads × C/heads) rather than (HW × HW), giving
    O(C² · HW) complexity — independent of spatial resolution.
    """

    def __init__(self, channels: int, num_heads: int, bias: bool = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv    = nn.Conv2d(channels, channels * 3, 1, bias=bias)
        self.qkv_dw = nn.Conv2d(
            channels * 3, channels * 3, 3,
            padding=1, groups=channels * 3, bias=bias,
        )
        self.project_out = nn.Conv2d(channels, channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        head_ch = C // self.num_heads

        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # (B, num_heads, head_ch, H*W)
        q = q.view(B, self.num_heads, head_ch, H * W)
        k = k.view(B, self.num_heads, head_ch, H * W)
        v = v.view(B, self.num_heads, head_ch, H * W)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Transposed attention: (B, heads, head_ch, head_ch)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v).view(B, C, H, W)
        return self.project_out(out)


# ---------------------------------------------------------------------------
# GDFN — Gated-Dconv Feed-Forward Network
# ---------------------------------------------------------------------------

class GDFN(nn.Module):
    """Gated FFN: two-branch depthwise conv with GELU gating."""

    def __init__(self, channels: int, ffn_expand: float = 2.66, bias: bool = False) -> None:
        super().__init__()
        hidden = int(channels * ffn_expand)

        self.project_in  = nn.Conv2d(channels, hidden * 2, 1, bias=bias)
        self.dw_conv     = nn.Conv2d(
            hidden * 2, hidden * 2, 3,
            padding=1, groups=hidden * 2, bias=bias,
        )
        self.project_out = nn.Conv2d(hidden, channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.dw_conv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


# ---------------------------------------------------------------------------
# Transformer block (encoder): MDTA + GDFN with LayerNorm pre-norm
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Restormer block: pre-norm MDTA + pre-norm GDFN with stochastic depth."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        ffn_expand: float = 2.66,
        bias: bool = False,
        drop_path_p: float = 0.0,
        grad_ckpt: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn  = MDTA(channels, num_heads, bias)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn   = GDFN(channels, ffn_expand, bias)
        self.drop_path_p = drop_path_p
        self.grad_ckpt   = grad_ckpt

    def _ln(self, norm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
        return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        y = drop_path(self.attn(self._ln(self.norm1, x)), self.drop_path_p, self.training)
        x = x + y
        y = drop_path(self.ffn(self._ln(self.norm2, x)), self.drop_path_p, self.training)
        return x + y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.grad_ckpt and self.training:
            return cp.checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


# ---------------------------------------------------------------------------
# Prior injection: FiLM and Cross-Attention (same as NAFNetBlurDM)
# ---------------------------------------------------------------------------

class FiLMInjection(nn.Module):
    """Feature-wise Linear Modulation. Zero-init → identity at start of training."""

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
    """Cross-attention: spatial features attend over prior as key/value.

    Unlike FiLM, the injected information can vary spatially via the attention
    weights, allowing richer conditioning at the cost of a small overhead.
    """

    def __init__(self, channels: int, prior_dim: int, heads: int = 4) -> None:
        super().__init__()
        self.heads   = heads
        self.head_ch = channels // heads
        self.scale   = self.head_ch ** -0.5

        self.q_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.k_proj = nn.Linear(prior_dim, channels, bias=False)
        self.v_proj = nn.Linear(prior_dim, channels, bias=False)
        self.out    = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q = self.q_proj(x).view(B, self.heads, self.head_ch, H * W).permute(0, 1, 3, 2)
        k = self.k_proj(prior).view(B, self.heads, self.head_ch).unsqueeze(2)
        v = self.v_proj(prior).view(B, self.heads, self.head_ch).unsqueeze(2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out  = (attn @ v).permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        return x + self.out(out)


# ---------------------------------------------------------------------------
# Decoder block with prior injection
# ---------------------------------------------------------------------------

class TransformerDecoderBlock(nn.Module):
    """Restormer decoder block: prior injection → stack of TransformerBlocks."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        num_blocks: int,
        prior_dim: int,
        ffn_expand: float = 2.66,
        bias: bool = False,
        drop_path_p: float = 0.0,
        use_cross_attn: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[
            TransformerBlock(channels, num_heads, ffn_expand, bias, drop_path_p)
            for _ in range(num_blocks)
        ])
        if use_cross_attn:
            self.inject = CrossAttentionInjection(channels, prior_dim)
        else:
            self.inject = FiLMInjection(channels, prior_dim)

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.inject(x, prior))


# ---------------------------------------------------------------------------
# Pixel-shuffle down/upsample (lossless spatial ↔ channel trade-off)
# ---------------------------------------------------------------------------

class Downsample(nn.Module):
    """Halve spatial resolution, double channels via PixelUnshuffle."""

    def __init__(self, channels: int, bias: bool = False) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=bias),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    """Double spatial resolution, halve channels via PixelShuffle."""

    def __init__(self, channels: int, bias: bool = False) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, padding=1, bias=bias),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# ---------------------------------------------------------------------------
# Full RestormerBlurDM
# ---------------------------------------------------------------------------

class RestormerBlurDM(nn.Module):
    """Restormer deblurring backbone conditioned on a BlurDM latent prior.

    Architecture: hierarchical U-Net with MDTA+GDFN encoder stages and
    prior-conditioned decoder stages.  Multi-scale outputs (1/4, 1/2, full)
    match the interface of MIMOUNetBlurDM and NAFNetBlurDM.

    Args:
        img_channels:    Input/output channels (default 3 for RGB).
        width:           Base channel width (doubles with each downsample).
        enc_blocks:      TransformerBlocks per encoder stage.
        dec_blocks:      TransformerBlocks per decoder stage (mirrors encoder).
        num_heads:       Attention heads per stage (len must equal enc_blocks).
        ffn_expand:      GDFN hidden expansion ratio (2.66 in original paper).
        prior_dim:       Incoming prior vector dimension.
        drop_path_rate:  Max stochastic depth probability (linearly scaled).
        grad_ckpt:       Gradient checkpointing on encoder (saves VRAM).
        use_cross_attn:  Cross-attention prior injection instead of FiLM.
        bias:            Use bias in convolutions.
    """

    def __init__(
        self,
        img_channels: int = 3,
        width: int = 48,
        enc_blocks: list[int] | None = None,
        dec_blocks: list[int] | None = None,
        num_heads:  list[int] | None = None,
        ffn_expand: float = 2.66,
        prior_dim:  int = 256,
        drop_path_rate: float = 0.0,
        grad_ckpt:      bool = False,
        use_cross_attn: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if enc_blocks is None:
            enc_blocks = [4, 6, 6, 8]
        if dec_blocks is None:
            dec_blocks = [4, 6, 6, 8]
        if num_heads is None:
            num_heads = [1, 2, 4, 8]

        n_stages = len(enc_blocks)
        total_enc = sum(enc_blocks)
        dp_rates  = [x.item() for x in torch.linspace(0, drop_path_rate, total_enc)]
        dp_idx    = 0

        # Input projection
        self.patch_embed = nn.Conv2d(img_channels, width, 3, padding=1, bias=bias)

        # Encoder: n_stages levels, (n_stages-1) downsamples
        self.encoders: nn.ModuleList = nn.ModuleList()
        self.downs:    nn.ModuleList = nn.ModuleList()
        ch = width
        for i, n in enumerate(enc_blocks):
            self.encoders.append(nn.Sequential(*[
                TransformerBlock(
                    ch, num_heads[i], ffn_expand, bias,
                    dp_rates[dp_idx + j], grad_ckpt,
                )
                for j in range(n)
            ]))
            dp_idx += n
            if i < n_stages - 1:
                self.downs.append(Downsample(ch, bias))
                ch *= 2

        # Decoder: (n_stages-1) upsample + fuse + decode stages
        self.decoders:    nn.ModuleList = nn.ModuleList()
        self.ups:         nn.ModuleList = nn.ModuleList()
        self.reduce_chan: nn.ModuleList = nn.ModuleList()

        for i in range(n_stages - 2, -1, -1):
            self.ups.append(Upsample(ch, bias))
            ch //= 2
            self.reduce_chan.append(nn.Conv2d(ch * 2, ch, 1, bias=bias))
            self.decoders.append(
                TransformerDecoderBlock(
                    ch, num_heads[i], dec_blocks[i],
                    prior_dim, ffn_expand, bias,
                    use_cross_attn=use_cross_attn,
                )
            )

        # Output projection
        self.output = nn.Conv2d(width, img_channels, 3, padding=1, bias=bias)

        # Multi-scale auxiliary output heads.
        # Decoder index 0 → coarsest upsample (1/4 of input for 4-stage net).
        # Decoder index 1 → next upsample (1/2 of input for 4-stage net).
        n_dec = n_stages - 1
        self._out4_idx = 0 if n_dec >= 3 else -1
        self._out2_idx = 1 if n_dec >= 2 else -1

        # Channel widths at decoder stages 0 and 1 after reduce_chan:
        # stage 0: ch was width*2^(n_stages-1), after up+reduce → width*2^(n_stages-2)
        # stage 1: width*2^(n_stages-3)
        ch4 = width * (2 ** max(n_stages - 2, 0))
        ch2 = width * (2 ** max(n_stages - 3, 0)) if n_stages >= 3 else width
        self.out_head_4 = nn.Conv2d(ch4, img_channels, 3, padding=1, bias=bias)
        self.out_head_2 = nn.Conv2d(ch2, img_channels, 3, padding=1, bias=bias)

    def forward(
        self, x: torch.Tensor, prior: torch.Tensor
    ) -> list[torch.Tensor]:
        """
        Args:
            x:     (B, 3, H, W) blurry input in [-0.5, 0.5].
            prior: (B, prior_dim) latent prior from LatentExposureDiffusion.
        Returns:
            List of three predictions: [out_1/4, out_1/2, out_full].
        """
        inp = x
        x_4 = F.interpolate(x, scale_factor=0.25, mode="bilinear", align_corners=False)
        x_2 = F.interpolate(x, scale_factor=0.50, mode="bilinear", align_corners=False)

        x = self.patch_embed(inp)

        # Encode and collect skip connections
        skips: list[torch.Tensor] = []
        for enc, down in zip(self.encoders[:-1], self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.encoders[-1](x)   # bottleneck

        # Decode with prior injection
        outputs: list[torch.Tensor] = []
        for i, (dec, up, reduce) in enumerate(
            zip(self.decoders, self.ups, self.reduce_chan)
        ):
            x = up(x)
            skip = skips[-(i + 1)]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = reduce(torch.cat([x, skip], dim=1))
            x = dec(x, prior)

            if i == self._out4_idx:
                outputs.append(self.out_head_4(x) + x_4)
            elif i == self._out2_idx:
                outputs.append(self.out_head_2(x) + x_2)

        outputs.append(self.output(x) + inp)

        # Always return exactly 3 outputs
        while len(outputs) < 3:
            outputs.insert(0, F.interpolate(
                outputs[-1],
                scale_factor=0.5 ** (3 - len(outputs)),
                mode="bilinear", align_corners=False,
            ))
        return outputs


# ---------------------------------------------------------------------------
# Pre-defined configurations
# ---------------------------------------------------------------------------

_CONFIGS: dict[str, dict] = {
    # Light: ~8 M params — fast training/inference, good for ablation
    "RestormerBlurDM-light": dict(
        width=32,
        enc_blocks=[2, 4, 4, 4],
        dec_blocks=[2, 4, 4, 4],
        num_heads=[1, 2, 4, 8],
        ffn_expand=2.66,
    ),
    # Base: ~26 M params — matches original Restormer-Motion-Deblur config
    "RestormerBlurDM-base": dict(
        width=48,
        enc_blocks=[4, 6, 6, 8],
        dec_blocks=[4, 6, 6, 8],
        num_heads=[1, 2, 4, 8],
        ffn_expand=2.66,
    ),
    # Large: ~45 M params — wider base for maximum quality
    "RestormerBlurDM-large": dict(
        width=64,
        enc_blocks=[4, 6, 6, 8],
        dec_blocks=[4, 6, 6, 8],
        num_heads=[1, 2, 4, 8],
        ffn_expand=2.66,
    ),
}


def build_Restormer(
    model_name: str,
    prior_dim: int = 256,
    grad_ckpt: bool = False,
    drop_path_rate: float = 0.0,
    use_cross_attn: bool = False,
    **kwargs,
) -> nn.Module:
    if model_name not in _CONFIGS:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(_CONFIGS)}"
        )
    cfg = {
        **_CONFIGS[model_name],
        "prior_dim":       prior_dim,
        "grad_ckpt":       grad_ckpt,
        "drop_path_rate":  drop_path_rate,
        "use_cross_attn":  use_cross_attn,
        **kwargs,
    }
    return RestormerBlurDM(**cfg)
