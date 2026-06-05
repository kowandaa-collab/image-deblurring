# Presentation: Slide 2 & 3 (GoPro test set, 1111 images)

## Slide 2 — Results of the existing model

**Model:** MIMO-UNet + BlurDM (3-stage training from the BlurDM paper)

**What it is:** The original deblurring backbone used in the NeurIPS 2025 BlurDM work, with the latent blur-diffusion prior plugged into the network.

**Average metrics (GoPro test):**

| PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|--------|--------|---------|
| 25.69 | 0.857 | 0.282 |

**Image for slide:** `presentation/compare_GOPR0881_11_01_000287.png` — use columns **Blur** and **MIMO-UNet + BlurDM** (or crop that panel).

**Talking points:**
- Input is motion-blurred; model recovers structure but some edges stay soft.
- Trained with Stage 1 (encoder + backbone) → Stage 2 (BlurDM) → Stage 3 (joint fine-tuning).

---

## Slide 3 — Results of your modified model

**Model:** NAFNet + BlurDM (your implementation: NAFNetBlurDM-light + BlurDM-v4 training)

**How it differs from Slide 2:**

| Aspect | Existing (MIMO-UNet + BlurDM) | Your modified (NAFNet + BlurDM) |
|--------|-------------------------------|----------------------------------|
| Backbone | MIMO-UNet (multi-scale encoder–decoder) | NAFNet (NAF blocks, channel attention) |
| Prior | Latent BlurDM diffusion (dual noise + blur) | Same BlurDM prior in latent space |
| Training | Paper 3-stage pipeline | Same stages + v4 speed/quality options (AMP, cosine schedule, multi-scale LE) |
| Metrics | PSNR **25.69**, SSIM **0.857** | PSNR 25.57, SSIM 0.843, **LPIPS 0.280** (better perceptual score) |
| Optional | — | TTA: PSNR **25.66**, LPIPS **0.279** |

**Image for slide:** Same comparison file — columns **NAFNet + BlurDM** vs **MIMO-UNet + BlurDM**, or `compare_tta_*.png` for TTA.

**Example (GOPR0881_11_01_000287.png):** NAFNet PSNR 27.04 vs MIMO 26.75 on this frame — sharper building edges.

**Talking points:**
- Same BlurDM idea; you swapped the backbone to NAFNet and retrained on GoPro.
- NAFNet + BlurDM recovers finer detail on many scenes (lower LPIPS = closer to human perception).
- BlurDM still does dual denoising + deblurring in latent space (not plain single-network deblurring).

---

## Recommended comparison figures (ready to insert)

| File | Use on |
|------|--------|
| `compare_GOPR0881_11_01_000287.png` | Main outdoor scene — NAFNet visibly sharper |
| `compare_GOPR0385_11_01_003077.png` | Large PSNR gain for NAFNet on this frame |
| `compare_GOPR0881_11_01_000283.png` | Second strong NAFNet example |
| `compare_tta_GOPR0881_11_01_000287.png` | Optional: show TTA improvement |

Regenerate: `python make_slide_comparisons.py --images <filename>.png`

---

## BlurDM vs “plain” deblurring (paper context, 1 bullet)

The **paper** compares **Baseline** (backbone alone) vs **BlurDM** (backbone + diffusion prior). Your runs are **both BlurDM-enhanced**; Slide 2 vs 3 compares **two backbones** under BlurDM, not backbone-without-BlurDM.
