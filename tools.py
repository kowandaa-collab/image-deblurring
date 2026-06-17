"""
tools.py — BlurDM evaluation and visualization utilities.

Subcommands:
  eval    Compute PSNR / SSIM / LPIPS between a prediction folder and ground truth.
  compare Compare AVERAGE rows across multiple metrics CSV files.
  slides  Build side-by-side comparison figures for presentation slides.

Usage:
  python tools.py eval   --gt <gt_dir> --pred <pred_dir> [--output-csv out.csv]
  python tools.py compare --csv a.csv b.csv c.csv
  python tools.py slides  --images img1.png img2.png [--root .] [--data D:/dataset/test/GoPro]
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from torchvision import transforms

from src.utils.metrics.psnr_ssim import calculate_psnr, calculate_ssim


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

_to_lpips = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def cmd_eval(args: argparse.Namespace) -> None:
    import lpips as lpips_lib
    device = torch.device(args.device)
    lpips_model = lpips_lib.LPIPS(net=args.lpips_net).to(device).eval()

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

        gt_np, pred_np = np.array(gt_img), np.array(pred_img)
        psnr_val = calculate_psnr(gt_np, pred_np, crop_border=0, test_y_channel=False)
        ssim_val = calculate_ssim(gt_np, pred_np, crop_border=0, test_y_channel=False)

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
    print(f"  PSNR  (higher=better) : {avg_psnr:.4f} dB")
    print(f"  SSIM  (higher=better) : {avg_ssim:.4f}")
    print(f"  LPIPS (lower=better)  : {avg_lpips:.4f}")
    print(f"{'='*40}\n")

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "psnr", "ssim", "lpips"])
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow({"file": "AVERAGE", "psnr": avg_psnr, "ssim": avg_ssim, "lpips": avg_lpips})
        print(f"Per-image results saved to {args.output_csv}")


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def _read_average(csv_path: Path) -> dict[str, float]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("file") == "AVERAGE":
                return {"psnr": float(row["psnr"]), "ssim": float(row["ssim"]), "lpips": float(row["lpips"])}
    raise ValueError(f"No AVERAGE row found in {csv_path}")


def cmd_compare(args: argparse.Namespace) -> None:
    rows: list[dict] = []
    for csv_file in args.csv:
        p = Path(csv_file)
        avg = _read_average(p)
        label_parts = p.parts[-3:] if len(p.parts) >= 3 else p.parts
        rows.append({"name": "/".join(label_parts).replace(".csv", ""), "path": str(p), **avg})

    by_psnr  = sorted(rows, key=lambda r: r["psnr"],  reverse=True)
    by_ssim  = sorted(rows, key=lambda r: r["ssim"],  reverse=True)
    by_lpips = sorted(rows, key=lambda r: r["lpips"])

    print("\n=== Metrics Comparison ===")
    print(f"{'Run':35} {'PSNR':>9} {'SSIM':>9} {'LPIPS':>9}")
    print("-" * 66)
    for r in rows:
        print(f"{str(r['name']):35} {float(r['psnr']):9.4f} {float(r['ssim']):9.4f} {float(r['lpips']):9.4f}")

    print("\nBest by metric:")
    print(f"  PSNR : {by_psnr[0]['name']} ({float(by_psnr[0]['psnr']):.4f})")
    print(f"  SSIM : {by_ssim[0]['name']} ({float(by_ssim[0]['ssim']):.4f})")
    print(f"  LPIPS: {by_lpips[0]['name']} ({float(by_lpips[0]['lpips']):.4f})")

    print("\nCSV inputs:")
    for r in rows:
        print(f"  {r['path']}")


# ---------------------------------------------------------------------------
# slides
# ---------------------------------------------------------------------------

def _load_rgb(path: Path, max_w: int = 640) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if img.width > max_w:
        h = int(img.height * max_w / img.width)
        img = img.resize((max_w, h), Image.Resampling.LANCZOS)
    return img


def _pad_height(images: list[Image.Image]) -> list[Image.Image]:
    h = max(im.height for im in images)
    out = []
    for im in images:
        if im.height == h:
            out.append(im)
        else:
            canvas = Image.new("RGB", (im.width, h), (32, 32, 32))
            canvas.paste(im, (0, (h - im.height) // 2))
            out.append(canvas)
    return out


def _hstack(images: list[Image.Image], labels: list[str], gap: int = 8) -> Image.Image:
    images = _pad_height(images)
    font = ImageFont.load_default()
    label_h = 28
    total_w = sum(im.width for im in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (total_w, images[0].height + label_h), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for im, lab in zip(images, labels):
        canvas.paste(im, (x, label_h))
        draw.text((x + 6, 6), lab, fill=(255, 255, 255), font=font)
        x += im.width + gap
    return canvas


def cmd_slides(args: argparse.Namespace) -> None:
    root    = args.root
    out_dir = args.out or root / "presentation"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.images:
        blur    = args.data / "input"  / name
        sharp   = args.data / "target" / name
        mimo    = root / "results/MIMO_UNet/GoPro/GoPro" / name
        naf     = root / "results/NAFNet/GoPro/GoPro" / name
        naf_tta = root / "results/NAFNet/GoPro_tta/GoPro" / name

        paths  = [blur, sharp, mimo, naf]
        labels = ["Blur (input)", "Sharp (GT)", "MIMO-UNet + BlurDM", "NAFNet + BlurDM"]

        if not all(p.exists() for p in paths):
            print(f"Skip {name}: missing {[str(p) for p in paths if not p.exists()]}")
            continue

        row = _hstack([_load_rgb(p, args.max_w) for p in paths], labels)
        out_path = out_dir / f"compare_{name}"
        row.save(out_path)
        print(f"Wrote {out_path}")

        if naf_tta.exists():
            row2 = _hstack(
                [_load_rgb(blur, args.max_w), _load_rgb(mimo, args.max_w),
                 _load_rgb(naf,  args.max_w), _load_rgb(naf_tta, args.max_w)],
                ["Blur", "MIMO + BlurDM", "NAFNet + BlurDM", "NAFNet + TTA"],
            )
            row2.save(out_dir / f"compare_tta_{name}")
            print(f"Wrote {out_dir / f'compare_tta_{name}'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BlurDM evaluation and visualization tools.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # -- eval --
    e = sub.add_parser("eval", help="Compute PSNR/SSIM/LPIPS between two folders.")
    e.add_argument("--gt",         required=True, help="Ground-truth image folder.")
    e.add_argument("--pred",       required=True, help="Predicted image folder.")
    e.add_argument("--lpips-net",  default="alex", choices=["alex", "vgg", "squeeze"])
    e.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    e.add_argument("--output-csv", default=None,  help="Optional CSV for per-image results.")

    # -- compare --
    c = sub.add_parser("compare", help="Compare AVERAGE rows across metrics CSV files.")
    c.add_argument("--csv", nargs="+", required=True, help="Metrics CSV paths.")

    # -- slides --
    s = sub.add_parser("slides", help="Build side-by-side slide comparison figures.")
    s.add_argument("--root",    type=Path, default=Path(__file__).resolve().parent)
    s.add_argument("--data",    type=Path, default=Path("D:/dataset/test/GoPro"))
    s.add_argument("--out",     type=Path, default=None)
    s.add_argument("--images",  nargs="+", required=True)
    s.add_argument("--max-w",   type=int,  default=480)

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    {"eval": cmd_eval, "compare": cmd_compare, "slides": cmd_slides}[args.cmd](args)
