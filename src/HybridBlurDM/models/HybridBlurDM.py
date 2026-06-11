"""
HybridBlurDM: Custom hybrid architecture for SOTA motion deblurring.

Design goals
============
  * Outperform NAFNet (31.71 dB Stage-3 GoPro) by combining the best ideas
    from recent literature into a single, parameter-efficient backbone.

Key innovations vs. existing backbones in this project
=======================================================
  1. DualDomainMixer (per-block):
     FFT branch (global receptive field, O(HW log HW)) and depthwise-spatial
     branch (local, NAFNet-style SCA) run in PARALLEL and are fused with a
     learned channel-wise gate.  Compare to FFTformer (sequential: FFT then
     spatial) and NAFNet (spatial only).

  2. MotionStripConv (per-block):
     Asymmetric depthwise convolutions (1×K horizontal + K×1 vertical) capture
     directional motion-blur streaks cheaply.  Inspired by Stripformer but
     replaced with efficient DW strip convs to avoid O(HW²) strip attention.
     Channel attention on the merged H+V feature provides global directionality.
     K=21 covers longer streak lengths at 256px training resolution.

  3. GatedFFN (SimpleGate, per-block):
     NAFNet-style element-wise gate (a * b from split, no GELU activation)
     instead of Restormer's GELU-gated FFN — better gradient flow.

  4. FiLM prior injection at EVERY decoder level, not just the bottleneck.
     This gives the diffusion prior four distinct opportunities to steer the
     reconstruction at 1/4, 1/2, and full scale, yielding larger Stage-3 gains.

  5. Deformable convolution at bottleneck encoder blocks (use_deform=True):
     torchvision.ops.DeformConv2d replaces the depthwise spatial conv in
     _SpatialBranch for bottleneck encoder blocks only.  Offsets are zero-init
     so the block starts as a standard convolution and learns to deform.
     Falls back to a spatial-attention gate if DeformConv2d is unavailable.

Architecture summary
====================
  U-Net (4 stages, PixelShuffle down/up):
    patch_embed → [enc0→enc1→enc2→enc3(bottleneck)] → [dec0→dec1→dec2→output]
  Multi-scale outputs: [pred_1/4, pred_1/2, pred_full]  (same as NAFNet/Restormer)

Estimated params (HybridBlurDM-light, width=32, enc=[2,2,4,8]):
  Backbone ≈ 18–22 M  |  LE_arch ≈ 5.27 M  |  Total ≈ 24–27 M
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


# ---------------------------------------------------------------------------
# Optional deformable convolution (torchvision.ops.DeformConv2d)
# ---------------------------------------------------------------------------

try:
    from torchvision.ops import DeformConv2d as _DeformConv2d
    _DEFORM_AVAILABLE = True
except (ImportError, AttributeError):
    _DeformConv2d = None
    _DEFORM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep  = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask  = torch.empty(shape, device=x.device).bernoulli_(keep).div_(keep)
    return x * mask


class LayerNorm2d(nn.Module):
    """Channel-first (B, C, H, W) layer norm via permute."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# SimpleChannelAttention — NAFNet-style (no reduction ratio, no softmax)
# ---------------------------------------------------------------------------

class SimpleChannelAttention(nn.Module):
    """Global average pool → 1×1 conv → sigmoid scale.

    Simpler and faster than SE-block — omits the bottleneck and ReLU, keeping
    only a single pointwise transform. This is the 'SCA' from NAFNet.
    """

    def __init__(self, dim: int, bias: bool = False) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.proj(self.pool(x))


# ---------------------------------------------------------------------------
# DualDomainMixer — parallel FFT + Spatial, learned gate fusion
# ---------------------------------------------------------------------------

class _FreqBranch(nn.Module):
    """Frequency-domain mixing on pre-normed input.

    Receives already-normalised features y, operates in float32 to avoid
    ComplexHalf (float16 complex is experimental in PyTorch), and casts back.
    Two 1×1 convs on the (H × W//2+1) frequency map give every spatial
    position a global receptive field at O(HW log HW) cost.
    """

    def __init__(self, dim: int, bias: bool = False) -> None:
        super().__init__()
        self.proj1 = nn.Conv2d(dim * 2, dim * 2, 1, bias=bias)
        self.proj2 = nn.Conv2d(dim * 2, dim * 2, 1, bias=bias)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        B, C, H, W = y.shape
        dtype = y.dtype
        # float16 complex (ComplexHalf) is experimental; promote to float32.
        yf    = y.float()
        fft   = torch.fft.rfft2(yf, norm="ortho")             # B, C, H, W//2+1 complex64
        ri    = torch.cat([fft.real, fft.imag], dim=1)         # B, 2C, H, W//2+1
        ri    = self.proj2(F.gelu(self.proj1(ri.to(dtype)))).float()
        yr, yi = ri.chunk(2, dim=1)
        return torch.fft.irfft2(torch.complex(yr, yi), s=(H, W), norm="ortho").to(dtype)


class _SpatialBranch(nn.Module):
    """Depthwise-3×3 + NAFNet-style SCA on pre-normed input.

    When use_deform=True and torchvision DeformConv2d is available, replaces
    the depthwise conv with a deformable conv whose offsets are zero-initialised
    (identity at init).  Falls back to a spatial-attention gate otherwise.
    """

    def __init__(self, dim: int, bias: bool = False, use_deform: bool = False) -> None:
        super().__init__()
        self._deform_active   = use_deform and _DEFORM_AVAILABLE
        self._deform_fallback = use_deform and not _DEFORM_AVAILABLE

        if self._deform_active:
            # Offset conv: predicts 2*kH*kW=18 offset channels for a 3×3 kernel
            self.offset_conv = nn.Conv2d(dim, 18, 3, padding=1, bias=True)
            nn.init.zeros_(self.offset_conv.weight)
            nn.init.zeros_(self.offset_conv.bias)
            self.dw = _DeformConv2d(dim, dim, 3, padding=1, bias=bias)
        elif self._deform_fallback:
            # Spatial attention gate as fallback when DeformConv2d is unavailable
            self.dw       = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=bias)
            self.attn_gate = nn.Sequential(
                nn.Conv2d(dim, dim, 1, bias=False),
                nn.Sigmoid(),
            )
        else:
            self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=bias)

        self.sca = SimpleChannelAttention(dim, bias)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if self._deform_active:
            offset = self.offset_conv(y)
            return self.sca(self.dw(y, offset))
        elif self._deform_fallback:
            return self.sca(self.dw(y) * self.attn_gate(y))
        else:
            return self.sca(self.dw(y))


class DualDomainMixer(nn.Module):
    """Single pre-norm, parallel FFT + Spatial, fused with a learned gate.

    Gate takes cat([freq_out, spatial_out]) → 1×1 conv → C channels.
    Learns adaptively when to rely on global frequency context vs. local
    spatial patterns — content-dependent mixing without fixed scalar weights.
    """

    def __init__(self, dim: int, bias: bool = False, use_deform: bool = False) -> None:
        super().__init__()
        self.norm    = LayerNorm2d(dim)
        self.freq    = _FreqBranch(dim, bias)
        self.spatial = _SpatialBranch(dim, bias, use_deform=use_deform)
        self.gate    = nn.Conv2d(dim * 2, dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        f = self.freq(y)
        s = self.spatial(y)
        return self.gate(torch.cat([f, s], dim=1))


# ---------------------------------------------------------------------------
# MotionStripConv — directional blur capture via asymmetric DW convs
# ---------------------------------------------------------------------------

class MotionStripConv(nn.Module):
    """Asymmetric depthwise convolutions for motion-blur directionality.

    1×K horizontal DW + K×1 vertical DW convs capture streak-shaped blur
    patterns at negligible parameter cost (K×C params each, no weight sharing
    across channels needed since blur direction may vary per channel group).
    Channel attention on the merged H+V feature highlights the dominant
    direction globally.

    Physical motivation: GoPro motion blur has typical streak lengths of
    K≈21 pixels at 256×256 training resolution.  Both H and V streaks are
    modelled simultaneously, with the projection learning to combine them.
    """

    def __init__(self, dim: int, kernel: int = 21, bias: bool = False) -> None:
        super().__init__()
        self.norm  = LayerNorm2d(dim)
        # asymmetric DW convs — capture directional context
        self.h_dw  = nn.Conv2d(dim, dim, (1, kernel), padding=(0, kernel // 2),
                               groups=dim, bias=bias)
        self.v_dw  = nn.Conv2d(dim, dim, (kernel, 1), padding=(kernel // 2, 0),
                               groups=dim, bias=bias)
        self.proj  = nn.Conv2d(dim * 2, dim, 1, bias=bias)
        self.sca   = SimpleChannelAttention(dim, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        h = self.h_dw(y)
        v = self.v_dw(y)
        return self.sca(self.proj(torch.cat([h, v], dim=1)))


# ---------------------------------------------------------------------------
# GatedFFN — NAFNet-style SimpleGate (element-wise, no GELU)
# ---------------------------------------------------------------------------

class GatedFFN(nn.Module):
    """Channel-mixing FFN with SimpleGate activation.

    SimpleGate: split channels in half → a * b (element-wise product).
    This replaces GELU with a multiplicative gate, giving better gradient
    flow and matching NAFNet's proven design.  A 3×3 depthwise conv before
    the gate provides spatial blending.
    """

    def __init__(self, dim: int, ffn_expand: float = 2.66, bias: bool = False) -> None:
        super().__init__()
        hidden       = int(dim * ffn_expand)
        self.norm    = LayerNorm2d(dim)
        self.proj_in  = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dw       = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1,
                                  groups=hidden * 2, bias=bias)
        self.proj_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y        = self.dw(self.proj_in(self.norm(x)))
        y1, y2   = y.chunk(2, dim=1)
        return self.proj_out(y1 * y2)   # SimpleGate: multiply, no sigmoid/gelu


# ---------------------------------------------------------------------------
# HybridBlock — the core repeating unit
# ---------------------------------------------------------------------------

class HybridBlock(nn.Module):
    """Core HybridBlurDM block.

    Three residual sub-steps, in sequence:
      1. DualDomainMixer  — global freq + local DW-SCA (or deformable), fused adaptively
      2. MotionStripConv  — horizontal + vertical directional context
      3. GatedFFN         — channel mixing via SimpleGate

    Each sub-step is wrapped in stochastic depth for regularisation.
    Gradient checkpointing is applied to the whole block (one recompute per
    block forward, saves the intermediate activations of all three steps).
    """

    def __init__(
        self,
        dim:          int,
        ffn_expand:   float = 2.66,
        strip_kernel: int   = 21,
        bias:         bool  = False,
        drop_path_p:  float = 0.0,
        grad_ckpt:    bool  = False,
        use_deform:   bool  = False,
    ) -> None:
        super().__init__()
        self.mixer       = DualDomainMixer(dim, bias, use_deform=use_deform)
        self.strip       = MotionStripConv(dim, strip_kernel, bias)
        self.ffn         = GatedFFN(dim, ffn_expand, bias)
        self.drop_path_p = drop_path_p
        self.grad_ckpt   = grad_ckpt

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = x + drop_path(self.mixer(x), self.drop_path_p, self.training)
        x = x + drop_path(self.strip(x), self.drop_path_p, self.training)
        x = x + drop_path(self.ffn(x),   self.drop_path_p, self.training)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.grad_ckpt and self.training:
            return cp.checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


# ---------------------------------------------------------------------------
# Prior injection (identical interface to FFTformerBlurDM / RestormerBlurDM)
# ---------------------------------------------------------------------------

class FiLMInjection(nn.Module):
    """Feature-wise Linear Modulation, zero-init → identity at training start."""

    def __init__(self, channels: int, prior_dim: int) -> None:
        super().__init__()
        self.film = nn.Linear(prior_dim, channels * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.ones_(self.film.bias[:channels])
        nn.init.zeros_(self.film.bias[channels:])

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        film          = self.film(prior).unsqueeze(-1).unsqueeze(-1)
        scale, shift  = film.chunk(2, dim=1)
        return x * scale + shift


class CrossAttentionInjection(nn.Module):
    """Cross-attention prior injection (optional richer alternative to FiLM)."""

    def __init__(self, channels: int, prior_dim: int, heads: int = 4) -> None:
        super().__init__()
        self.heads   = heads
        self.head_ch = channels // heads
        self.scale   = self.head_ch ** -0.5
        self.q_proj  = nn.Conv2d(channels, channels, 1, bias=False)
        self.k_proj  = nn.Linear(prior_dim, channels, bias=False)
        self.v_proj  = nn.Linear(prior_dim, channels, bias=False)
        self.out     = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q    = self.q_proj(x).view(B, self.heads, self.head_ch, H * W).permute(0, 1, 3, 2)
        k    = self.k_proj(prior).view(B, self.heads, self.head_ch).unsqueeze(2)
        v    = self.v_proj(prior).view(B, self.heads, self.head_ch).unsqueeze(2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out  = (attn @ v).permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        return x + self.out(out)


# ---------------------------------------------------------------------------
# Decoder block
# ---------------------------------------------------------------------------

class HybridDecoderBlock(nn.Module):
    """Prior injection → stack of HybridBlocks.

    FiLM injection at every decoder level (not just bottleneck) provides
    per-scale prior conditioning:  each decoder step can adapt the
    reconstruction at its own resolution based on the latent prior.
    Decoder blocks do not use deformable conv (bottleneck encoder only).
    """

    def __init__(
        self,
        dim:            int,
        num_blocks:     int,
        prior_dim:      int,
        ffn_expand:     float = 2.66,
        strip_kernel:   int   = 21,
        bias:           bool  = False,
        drop_path_p:    float = 0.0,
        use_cross_attn: bool  = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[
            HybridBlock(dim, ffn_expand, strip_kernel, bias, drop_path_p)
            for _ in range(num_blocks)
        ])
        if use_cross_attn:
            self.inject: nn.Module = CrossAttentionInjection(dim, prior_dim)
        else:
            self.inject = FiLMInjection(dim, prior_dim)

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.inject(x, prior))


# ---------------------------------------------------------------------------
# Pixel-shuffle down / upsample (identical to FFTformerBlurDM)
# ---------------------------------------------------------------------------

class Downsample(nn.Module):
    def __init__(self, channels: int, bias: bool = False) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=bias),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, channels: int, bias: bool = False) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, padding=1, bias=bias),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class HybridBlurDM(nn.Module):
    """Hybrid FFT-Spatial-Strip U-Net for motion deblurring.

    Interface is drop-in compatible with NAFNetBlurDM / RestormerBlurDM:
      forward(blur: B×3×H×W, prior: B×prior_dim) → [pred_1/4, pred_1/2, pred_full]

    Args:
        img_channels:   Input/output channels (3 for RGB).
        width:          Base channel width; doubled at each downsampling.
        enc_blocks:     HybridBlocks per encoder stage.
        dec_blocks:     HybridBlocks per decoder stage (mirrors enc).
        ffn_expand:     GatedFFN hidden expansion ratio.
        strip_kernel:   Asymmetric DW conv kernel length K (default 21).
        prior_dim:      LE_arch latent dim (256).
        drop_path_rate: Max stochastic-depth rate (linearly distributed).
        grad_ckpt:      Gradient checkpointing on encoder blocks.
        use_cross_attn: Cross-attention prior injection instead of FiLM.
        bias:           Bias in all convolutions.
        use_deform:     Enable deformable conv at bottleneck encoder blocks.
    """

    def __init__(
        self,
        img_channels:   int             = 3,
        width:          int             = 32,
        enc_blocks:     list[int] | None = None,
        dec_blocks:     list[int] | None = None,
        ffn_expand:     float           = 2.66,
        strip_kernel:   int             = 21,
        prior_dim:      int             = 256,
        drop_path_rate: float           = 0.0,
        grad_ckpt:      bool            = False,
        use_cross_attn: bool            = False,
        bias:           bool            = False,
        use_deform:     bool            = True,
    ) -> None:
        super().__init__()
        if enc_blocks is None:
            enc_blocks = [2, 2, 4, 8]
        if dec_blocks is None:
            dec_blocks = [2, 2, 4, 8]

        n_stages  = len(enc_blocks)
        total_enc = sum(enc_blocks)
        dp_rates  = [x.item() for x in torch.linspace(0, drop_path_rate, total_enc)]
        dp_idx    = 0

        self.patch_embed = nn.Conv2d(img_channels, width, 3, padding=1, bias=bias)

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoders: nn.ModuleList = nn.ModuleList()
        self.downs:    nn.ModuleList = nn.ModuleList()
        ch = width
        for i, n in enumerate(enc_blocks):
            is_bottleneck = (i == n_stages - 1)
            self.encoders.append(nn.Sequential(*[
                HybridBlock(ch, ffn_expand, strip_kernel, bias,
                            dp_rates[dp_idx + j], grad_ckpt,
                            use_deform=(is_bottleneck and use_deform))
                for j in range(n)
            ]))
            dp_idx += n
            if i < n_stages - 1:
                self.downs.append(Downsample(ch, bias))
                ch *= 2

        # ── Decoder ──────────────────────────────────────────────────────────
        self.decoders:    nn.ModuleList = nn.ModuleList()
        self.ups:         nn.ModuleList = nn.ModuleList()
        self.reduce_chan: nn.ModuleList = nn.ModuleList()

        for i in range(n_stages - 2, -1, -1):
            self.ups.append(Upsample(ch, bias))
            ch //= 2
            self.reduce_chan.append(nn.Conv2d(ch * 2, ch, 1, bias=bias))
            self.decoders.append(
                HybridDecoderBlock(
                    ch, dec_blocks[i], prior_dim,
                    ffn_expand, strip_kernel, bias,
                    drop_path_p=0.0,            # no drop_path in decoder
                    use_cross_attn=use_cross_attn,
                )
            )

        # ── Output heads ─────────────────────────────────────────────────────
        self.output = nn.Conv2d(width, img_channels, 3, padding=1, bias=bias)

        n_dec          = n_stages - 1
        self._out4_idx = 0 if n_dec >= 3 else -1
        self._out2_idx = 1 if n_dec >= 2 else -1

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
            prior: (B, prior_dim) latent prior from LE_arch.
        Returns:
            [pred_1/4, pred_1/2, pred_full]
        """
        inp = x
        x_4 = F.interpolate(x, scale_factor=0.25, mode="bilinear", align_corners=False)
        x_2 = F.interpolate(x, scale_factor=0.50, mode="bilinear", align_corners=False)

        x = self.patch_embed(inp)

        # Encode
        skips: list[torch.Tensor] = []
        for enc, down in zip(self.encoders[:-1], self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.encoders[-1](x)    # bottleneck

        # Decode
        outputs: list[torch.Tensor] = []
        for i, (dec, up, reduce) in enumerate(
            zip(self.decoders, self.ups, self.reduce_chan)
        ):
            x = up(x)
            skip = skips[-(i + 1)]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:],
                                  mode="bilinear", align_corners=False)
            x = reduce(torch.cat([x, skip], dim=1))
            x = dec(x, prior)

            if i == self._out4_idx:
                outputs.append(self.out_head_4(x) + x_4)
            elif i == self._out2_idx:
                outputs.append(self.out_head_2(x) + x_2)

        outputs.append(self.output(x) + inp)

        # Pad to always return exactly 3 outputs
        while len(outputs) < 3:
            outputs.insert(0, F.interpolate(
                outputs[-1],
                scale_factor=0.5 ** (3 - len(outputs)),
                mode="bilinear", align_corners=False,
            ))
        return outputs


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

_CONFIGS: dict[str, dict] = {
    # ~18-22 M backbone params. Mirrors NAFNet's proven [2,2,4,8] depth
    # structure but with 3× richer blocks (dual-domain + strip + gated FFN).
    # strip_kernel=21 covers longer GoPro streak lengths vs. the original 11.
    "HybridBlurDM-light": dict(
        width=32,
        enc_blocks=[2, 2, 4, 8],
        dec_blocks=[2, 2, 4, 8],
        ffn_expand=2.66,
        strip_kernel=21,
    ),
    # ~40 M backbone params — higher capacity for Stage-3 fine-tuning.
    "HybridBlurDM-base": dict(
        width=48,
        enc_blocks=[2, 2, 4, 8],
        dec_blocks=[2, 2, 4, 8],
        ffn_expand=2.66,
        strip_kernel=21,
    ),
}


def build_HybridBlurDM(
    model_name:     str,
    prior_dim:      int   = 256,
    grad_ckpt:      bool  = False,
    drop_path_rate: float = 0.0,
    use_cross_attn: bool  = False,
    use_deform:     bool  = True,
    **kwargs,
) -> nn.Module:
    if model_name not in _CONFIGS:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(_CONFIGS)}"
        )
    cfg = {
        **_CONFIGS[model_name],
        "prior_dim":      prior_dim,
        "grad_ckpt":      grad_ckpt,
        "drop_path_rate": drop_path_rate,
        "use_cross_attn": use_cross_attn,
        "use_deform":     use_deform,
        **kwargs,
    }
    return HybridBlurDM(**cfg)
