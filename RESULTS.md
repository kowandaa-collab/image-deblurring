# GoPro test-set evaluation results

Deblurred PNGs (1280×720) and per-image metrics on the prepared GoPro test split (`1111` images + header row in CSV).

| Run | Images | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|-----|--------|--------|--------|---------|
| MIMO_UNet/GoPro | `MIMO_UNet/GoPro/GoPro/` | 25.6931 | 0.8570 | 0.2821 |
| NAFNet/GoPro | `NAFNet/GoPro/GoPro/` | 25.5710 | 0.8429 | 0.2801 |
| NAFNet/GoPro + TTA | `NAFNet/GoPro_tta/GoPro/` | 25.6602 | 0.8490 | 0.2788 |
| NAFNet/GoPro_full | `NAFNet/GoPro_full/GoPro/` | (metrics not computed) | | |

Metrics CSVs: `*/metrics_gopro*.csv` (columns: `file`, `psnr`, `ssim`, `lpips`).

PNG outputs are stored with **Git LFS** (`results/**/*.png`).
