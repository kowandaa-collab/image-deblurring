# HybridBlurDM: A Diffusion-Guided Hybrid Model for Image Deblurring

**Charlie 1123532 · Alvin 1123530**

---

## Abstract

We present HybridBlurDM, a FiLM-conditioned image deblurring framework that integrates BlurDM diffusion priors into a hybrid restoration backbone without requiring iterative sampling at inference time. Building on the original BlurDM's dual-diffusion forward process, we design two lightweight model variants — NAFNetBlurDM-light and HybridBlurDM-light — trained using a 3-stage curriculum on the GoPro benchmark. HybridBlurDM-light introduces a novel DualDomainMixer block that jointly processes spatial and frequency-domain features, achieving 31.96 dB PSNR in standalone training — surpassing NAFNetBlurDM-light with the full DM prior (31.71 dB). MIMO-UNetBlurDM, completing all three training stages, achieves 32.38 dB.

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

| Stage | Description | Epochs | Crop |
|-------|-------------|--------|------|
| Stage 1 | Backbone + Sharp LatentEncoder prior | 300 | 256px |
| Stage 2 | BlurDM diffusion training (backbone frozen) | — | — |
| Stage 3 | Joint fine-tuning with BlurDM prior | 50 | 384px |

Loss: `L_Charb + 0.1 · L_FFT`  
Optimizer: AdamW (β=0.9, 0.999), cosine LR 3e-4 → 1e-7  
EMA decay: 0.999 · 4-way Test-Time Augmentation (+0.2–0.3 dB free)

---

## Why HybridBlurDM Did Not Complete 3 Stages

HybridBlurDM only completed standalone Stage 1 training. Repeated architectural bugs during development of DualDomainMixer (FFT shape mismatches), MotionStripConv (kernel dimension errors), and deformable attention (offset initialization issues) consumed significant time, leaving insufficient time for Stage 2 and Stage 3 training before the project deadline.

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
