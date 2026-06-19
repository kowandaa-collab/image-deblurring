# HybridBlurDM: A Diffusion-Guided Hybrid Model for Image Deblurring

**Charlie 1123532 · Alvin 1123530**

---

## Abstract

We present HybridBlurDM, a FiLM-conditioned image deblurring framework that integrates BlurDM diffusion priors into a hybrid restoration backbone without requiring iterative sampling at inference time. Building on the original BlurDM's dual-diffusion forward process, we design two lightweight model variants — NAFNetBlurDM-light and HybridBlurDM-light — trained using a 3-stage curriculum on the GoPro benchmark. HybridBlurDM-light introduces a novel DualDomainMixer block that jointly processes spatial and frequency-domain features, achieving 31.96 dB PSNR in standalone training — surpassing NAFNetBlurDM-light with the full DM prior (31.71 dB). MIMO-UNetBlurDM, completing all three training stages, achieves 32.38 dB.

---

## Research Question

Can image deblurring performance be improved by injecting BlurDM diffusion priors into a NAFNet-based restoration backbone via Feature-Wise Linear Modulation (FiLM)?

---

## Project Objectives

1. Study the BlurDM model and understand how diffusion priors are encoded.
2. Analyze how diffusion priors can guide a restoration backbone via FiLM conditioning.
3. Design and implement two model variants: NAFNetBlurDM-light and HybridBlurDM-light.
4. Implement the full 3-stage training curriculum: backbone → diffusion → joint fine-tuning.
5. Evaluate restoration performance using PSNR, SSIM, and LPIPS on the GoPro benchmark.

---

## HybridBlurDM-light Architecture

**HybridBlurDM-light** (~22M params) replaces each NAFNet encoder block with a custom hybrid block combining:

- **DualDomainMixer** — FFT global branch and depthwise spatial local branch run in parallel, fused with a learned channel-wise gate. Captures both frequency patterns (motion blur streaks) and spatial structure simultaneously.
- **MotionStripConv** — asymmetric depthwise convolutions (1×K horizontal + K×1 vertical) model directional motion blur cheaply, without the O(HW²) attention cost of Stripformer.
- **GatedFFN** — NAFNet-style element-wise gate (a × b from split channels) for clean gradient flow.
- **Deformable Bottleneck** — `DeformConv2d` at the encoder bottleneck with zero-initialized offsets, starting as standard conv and learning to deform to blur patterns.
- **Multi-scale FiLM injection** — BlurDM prior injected at every decoder level (four scales) via Feature-wise Linear Modulation: `FiLM(x) = γ·x + β`, where `[γ, β] = Linear(p)`.

---

## Results — GoPro Benchmark

| Model | Variant | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|-------|---------|--------|--------|---------|
| MIMO-UNetBlurDM | Stage 1 backbone | 32.23 dB | 0.919 (est.) | 0.1122 |
| MIMO-UNetBlurDM | Stage 3 + DM | **32.38 dB** | **0.924 (est.)** | **0.0980** |
| NAFNetBlurDM-light | Stage 1 backbone | 31.29 dB | 0.885 | 0.1318 |
| NAFNetBlurDM-light | Stage 3 + DM | 31.71 dB | 0.903 | 0.1193 |
| **HybridBlurDM-light** | **Standalone ★** | **31.96 dB** | **0.912** | **0.1129** |

★ HybridBlurDM-light standalone (no DM prior) already surpasses NAFNetBlurDM-light with the full DM prior by +0.25 dB.  
All metrics from crop-based validation. SSIM for MIMO-UNet estimated via linear extrapolation.

**Projected:** Full 3-stage HybridBlurDM training estimated at ~32.4–32.5 dB (based on +0.42 dB Stage 1→3 gain observed in NAFNet).

---

## 3-Stage Training Pipeline

| Stage | Description | Epochs | Crop | LR Range |
|-------|-------------|--------|------|----------|
| Stage 1 | Backbone + Sharp LatentEncoder prior | 300 | 256px | 3e-4 → 1e-7 |
| Stage 1b | Extended backbone training | +300 | 256px | 5e-5 → 1e-7 |
| Stage 2 | BlurDM diffusion training (backbone frozen) | 3000 | 256px | 1e-4 → 1e-6 |
| Stage 3 | Joint fine-tuning with BlurDM prior | 50 | 384px | 2e-6 → 1e-8 |

**Loss:** `L = L_Charbonnier + 0.1 · L_FFT`

The FFT loss computes Charbonnier error directly in the frequency domain:
```python
def fft_loss(criterion, pred, target):
    pf = torch.fft.rfft2(pred,   norm="backward")
    tf = torch.fft.rfft2(target, norm="backward")
    return criterion(
        torch.stack([pf.real, pf.imag], dim=-1),
        torch.stack([tf.real, tf.imag], dim=-1),
    )
```
This penalizes errors in the frequency spectrum directly, catching fine-texture and edge errors that pixel-space losses can underweight.

**Optimizer:** AdamW (β=0.9, 0.999), cosine LR schedule  
**EMA decay:** 0.999 · 4-way Test-Time Augmentation (+0.2–0.3 dB free)

---

## Why HybridBlurDM Did Not Complete 3 Stages

HybridBlurDM only completed standalone Stage 1 training. Four compounding issues prevented completion of Stage 2 and Stage 3:

- **High computational cost** — FFT operations and deformable attention are significantly more expensive than NAFNet's simple gated convolutions.
- **Long training time** — the extended Stage 1b run and the 3000-epoch Stage 2 requirement made the timeline tight.
- **Architectural bugs** — repeated tensor shape mismatches in DualDomainMixer (FFT frequency bin mismatch), MotionStripConv (kernel dimension errors), and deformable attention (offset initialization) required multiple rewrites.
- **Time lost to model errors** — the debugging time consumed the window available for Stage 2 and Stage 3 training before the project deadline.

---

## Based On

Original BlurDM paper:
```
@inproceedings{heblurdm,
  title={BlurDM: A Blur Diffusion Model for Image Deblurring},
  author={He, Jin-Ting and Tsai, Fu-Jen and Peng, Yan-Tsung and Chen, Min-Hung and Lin, Chia-Wen and Lin, Yen-Yu},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}
```
