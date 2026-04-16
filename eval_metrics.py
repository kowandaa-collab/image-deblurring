"""
eval_metrics.py — Compute PSNR, SSIM, and LPIPS between two image folders.

Improvements over the original:
  - CLI arguments via argparse (no hardcoded paths)
  - Correct LPIPS normalisation: images rescaled to [-1, 1] not [-0.5, 0.5]
  - PSNR/SSIM evaluated at native resolution (no forced 256×256 resize)
  - Progress bar via tqdm
  - Results printed as a formatted table and optionally saved to CSV
  - Graceful handling of unmatched filenames
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from src.utils.metrics.psnr_ssim import calculate_psnr, calculate_ssim


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate deblurring metrics.")
    p.add_argument("--gt",   required=True, help="Ground-truth image folder.")
    p.add_argument("--pred", required=True, help="Predicted image folder.")
    p.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-csv", default=None, help="Optional CSV to save per-image results.")
    return p.parse_args()


# Normalise to [-1, 1] for LPIPS (the model expects this range)
_to_lpips = transforms.Compose([
    transforms.ToTensor(),                                      # [0,1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1,1]
])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    lpips_model = lpips.LPIPS(net=args.lpips_net).to(device).eval()

    gt_files   = {f for f in os.listdir(args.gt)   if f.lower().endswith((".png", ".jpg", ".jpeg"))}
    pred_files = {f for f in os.listdir(args.pred)  if f.lower().endswith((".png", ".jpg", ".jpeg"))}
    common = sorted(gt_files & pred_files)

    if not common:
        print("No matching filenames found between the two folders.")
        return

    print(f"Evaluating {len(common)} image pairs…")

    rows: list[dict] = []

    for fname in tqdm(common):
        gt_img   = Image.open(os.path.join(args.gt,   fname)).convert("RGB")
        pred_img = Image.open(os.path.join(args.pred, fname)).convert("RGB")

        # Numpy arrays for PSNR / SSIM
        gt_np   = np.array(gt_img)
        pred_np = np.array(pred_img)

        psnr_val = calculate_psnr(gt_np, pred_np, crop_border=0, test_y_channel=False)
        ssim_val = calculate_ssim(gt_np, pred_np, crop_border=0, test_y_channel=False)

        # Tensors for LPIPS
        gt_t   = _to_lpips(gt_img).unsqueeze(0).to(device)
        pred_t = _to_lpips(pred_img).unsqueeze(0).to(device)
        with torch.no_grad():
            lpips_val = lpips_model(gt_t, pred_t).item()

        rows.append({"file": fname, "psnr": psnr_val, "ssim": ssim_val, "lpips": lpips_val})

    avg_psnr  = np.mean([r["psnr"]  for r in rows])
    avg_ssim  = np.mean([r["ssim"]  for r in rows])
    avg_lpips = np.mean([r["lpips"] for r in rows])

    print(f"\n{'='*40}")
    print(f"  Images evaluated : {len(rows)}")
    print(f"  PSNR  (↑ better) : {avg_psnr:.4f} dB")
    print(f"  SSIM  (↑ better) : {avg_ssim:.4f}")
    print(f"  LPIPS (↓ better) : {avg_lpips:.4f}")
    print(f"{'='*40}\n")

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "psnr", "ssim", "lpips"])
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow({"file": "AVERAGE", "psnr": avg_psnr, "ssim": avg_ssim, "lpips": avg_lpips})
        print(f"Per-image results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
