"""Build side-by-side figures for presentation slides (blur / sharp / MIMO / NAFNet)."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_rgb(path: Path, max_w: int = 640) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if img.width > max_w:
        h = int(img.height * max_w / img.width)
        img = img.resize((max_w, h), Image.Resampling.LANCZOS)
    return img


def pad_height(images: list[Image.Image]) -> list[Image.Image]:
    h = max(im.height for im in images)
    out = []
    for im in images:
        if im.height == h:
            out.append(im)
            continue
        canvas = Image.new("RGB", (im.width, h), (32, 32, 32))
        canvas.paste(im, (0, (h - im.height) // 2))
        out.append(canvas)
    return out


def hstack(images: list[Image.Image], labels: list[str], gap: int = 8) -> Image.Image:
    images = pad_height(images)
    font = ImageFont.load_default()
    label_h = 28
    total_w = sum(im.width for im in images) + gap * (len(images) - 1)
    total_h = images[0].height + label_h
    canvas = Image.new("RGB", (total_w, total_h), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for im, lab in zip(images, labels):
        canvas.paste(im, (x, label_h))
        draw.text((x + 6, 6), lab, fill=(255, 255, 255), font=font)
        x += im.width + gap
    return canvas


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--data", type=Path, default=Path("D:/dataset/test/GoPro"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--max-w", type=int, default=480)
    args = p.parse_args()
    root = args.root
    out_dir = args.out or root / "presentation"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.images:
        blur = args.data / "input" / name
        sharp = args.data / "target" / name
        mimo = root / "results/MIMO_UNet/GoPro/GoPro" / name
        naf = root / "results/NAFNet/GoPro/GoPro" / name
        naf_tta = root / "results/NAFNet/GoPro_tta/GoPro" / name
        paths = [blur, sharp, mimo, naf]
        labels = ["Blur (input)", "Sharp (GT)", "MIMO-UNet + BlurDM", "NAFNet + BlurDM"]
        if not all(p.exists() for p in paths):
            missing = [str(p) for p in paths if not p.exists()]
            print(f"Skip {name}: missing {missing}")
            continue
        row = hstack(
            [load_rgb(p, args.max_w) for p in paths],
            labels,
        )
        out_path = out_dir / f"compare_{name}"
        row.save(out_path)
        print(f"Wrote {out_path}")

        if naf_tta.exists():
            row2 = hstack(
                [
                    load_rgb(blur, args.max_w),
                    load_rgb(mimo, args.max_w),
                    load_rgb(naf, args.max_w),
                    load_rgb(naf_tta, args.max_w),
                ],
                ["Blur", "MIMO + BlurDM", "NAFNet + BlurDM", "NAFNet + TTA"],
            )
            p2 = out_dir / f"compare_tta_{name}"
            row2.save(p2)
            print(f"Wrote {p2}")


if __name__ == "__main__":
    main()
