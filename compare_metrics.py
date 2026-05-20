from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare multiple metrics CSV files.")
    p.add_argument(
        "--csv",
        nargs="+",
        required=True,
        help="One or more metrics CSV paths (must contain an AVERAGE row).",
    )
    return p.parse_args()


def read_average(csv_path: Path) -> dict[str, float]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("file") == "AVERAGE":
                return {
                    "psnr": float(row["psnr"]),
                    "ssim": float(row["ssim"]),
                    "lpips": float(row["lpips"]),
                }
    raise ValueError(f"No AVERAGE row found in {csv_path}")


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []

    for csv_file in args.csv:
        p = Path(csv_file)
        avg = read_average(p)
        label_parts = p.parts[-3:] if len(p.parts) >= 3 else p.parts
        rows.append(
            {
                "name": "/".join(label_parts).replace(".csv", ""),
                "path": str(p),
                **avg,
            }
        )

    by_psnr = sorted(rows, key=lambda r: r["psnr"], reverse=True)
    by_ssim = sorted(rows, key=lambda r: r["ssim"], reverse=True)
    by_lpips = sorted(rows, key=lambda r: r["lpips"])

    print("\n=== Metrics Comparison ===")
    print(f"{'Run':35} {'PSNR':>9} {'SSIM':>9} {'LPIPS':>9}")
    print("-" * 66)
    for r in rows:
        print(
            f"{str(r['name']):35} {float(r['psnr']):9.4f} {float(r['ssim']):9.4f} {float(r['lpips']):9.4f}"
        )

    print("\nBest by metric:")
    print(f"- PSNR : {by_psnr[0]['name']} ({float(by_psnr[0]['psnr']):.4f})")
    print(f"- SSIM : {by_ssim[0]['name']} ({float(by_ssim[0]['ssim']):.4f})")
    print(f"- LPIPS: {by_lpips[0]['name']} ({float(by_lpips[0]['lpips']):.4f})")

    print("\nCSV inputs:")
    for r in rows:
        print(f"- {r['path']}")


if __name__ == "__main__":
    main()
