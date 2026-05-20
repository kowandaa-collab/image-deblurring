# GoPro test-set evaluation results

Deblurred PNGs (1280×720) and per-image metrics on the prepared GoPro test split (`1111` images + header row in CSV).

| Run | Images | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|-----|--------|--------|--------|---------|
| MIMO_UNet/GoPro | `MIMO_UNet/GoPro/GoPro/` | 25.6931 | 0.8570 | 0.2821 |
| NAFNet/GoPro | `NAFNet/GoPro/GoPro/` | 25.5710 | 0.8429 | 0.2801 |
| NAFNet/GoPro + TTA | `NAFNet/GoPro_tta/GoPro/` | 25.6602 | 0.8490 | 0.2788 |
| NAFNet/GoPro_full | `NAFNet/GoPro_full/GoPro/` | (metrics not computed) | | |

Metrics CSVs (in repo): `results/MIMO_UNet/GoPro/metrics_gopro.csv`, `results/NAFNet/GoPro/metrics_gopro.csv`, `results/NAFNet/GoPro_tta/metrics_gopro_tta.csv` (columns: `file`, `psnr`, `ssim`, `lpips`).

Deblurred PNGs (~5 GB total) are **not** in git. Download from [GitHub Releases](https://github.com/kowandaa-collab/image-deblurring/releases/tag/gopro-results-v1) (after upload), or build zips locally:

```powershell
.\create_results_zips.ps1
gh auth login
.\upload_results_release.ps1
```

| Release asset | Contents |
|---------------|----------|
| `MIMO_UNet_GoPro.zip` | `results/MIMO_UNet/GoPro/GoPro/` (1111 PNGs) |
| `NAFNet_GoPro.zip` | `results/NAFNet/GoPro/GoPro/` |
| `NAFNet_GoPro_tta.zip` | `results/NAFNet/GoPro_tta/GoPro/` |
| `NAFNet_GoPro_full.zip` | `results/NAFNet/GoPro_full/GoPro/` |
