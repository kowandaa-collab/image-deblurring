"""
eval_tta.py — 4-way TTA evaluation for HybridBlurDM-light standalone checkpoint.

Usage:
    python src/HybridBlurDM/eval_tta.py \
        --ckpt experiments/HybridBlurDM/GoPro/standalone/best_HybridBlurDM-light.pth \
        --data_path D:/GOPRO_Large \
        --model_name HybridBlurDM-light

TTA: original + hflip + vflip + hvflip (4-way), outputs averaged before metrics.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from HybridBlurDM.models.HybridBlurDM import build_HybridBlurDM
from utils.utils import judge_and_remove_module_dict
from dataloader import Multi_GoPro_Loader
from torch.utils.data import DataLoader

import pyiqa


# ---------------------------------------------------------------------------

def tta_forward(model, blur, prior, amp: bool):
    """4-way sequential TTA: one forward pass per augmentation to avoid 4x VRAM spike."""
    from torch.amp import autocast

    augments = [
        (blur,                           lambda x: x),
        (torch.flip(blur, dims=[3]),     lambda x: torch.flip(x, dims=[3])),
        (torch.flip(blur, dims=[2]),     lambda x: torch.flip(x, dims=[2])),
        (torch.flip(blur, dims=[2, 3]),  lambda x: torch.flip(x, dims=[2, 3])),
    ]

    preds = []
    for x_aug, undo in augments:
        with autocast("cuda", enabled=amp):
            out = model(x_aug, prior)
        preds.append(undo(out[2].clamp(-0.5, 0.5).float()))

    return torch.stack(preds, dim=0).mean(dim=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",        required=True)
    p.add_argument("--data_path",   required=True)
    p.add_argument("--model_name",  default="HybridBlurDM-light")
    p.add_argument("--prior_dim",   type=int, default=256)
    p.add_argument("--amp",         action="store_true")
    p.add_argument("--no_tta",      action="store_true", help="Disable TTA (single-pass for comparison)")
    p.add_argument("--save_images", action="store_true")
    p.add_argument("--save_dir",    default="eval_output")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"TTA: {'disabled' if args.no_tta else '4-way (orig+hflip+vflip+hvflip)'}")

    # Build model
    net = build_HybridBlurDM(
        args.model_name,
        prior_dim=args.prior_dim,
        grad_ckpt=False,
        drop_path_rate=0.0,
        use_cross_attn=False,
        use_deform=True,
    ).to(device)

    # Load checkpoint
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    model_state = state.get("model_state", state)
    net.load_state_dict(judge_and_remove_module_dict(model_state))
    net.eval()
    print(f"Loaded checkpoint (epoch {state.get('epoch', '?')}, best_psnr={state.get('best_psnr', state.get('args', {}).best_psnr if hasattr(state.get('args', {}), 'best_psnr') else '?')})")

    # Dataset
    dataset = Multi_GoPro_Loader(args.data_path, mode="test", crop_size=None)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    print(f"Test set: {len(dataset)} images")

    # LPIPS only — PSNR computed inline as pure math (fast)
    lpips_fn = pyiqa.create_metric("lpips", device=device)

    if args.save_images:
        os.makedirs(args.save_dir, exist_ok=True)

    psnr_list, lpips_sample = [], []
    lpips_sample_every = max(1, len(dataset) // 100)  # ~100 LPIPS samples

    with torch.no_grad():
        for i, sample in enumerate(tqdm.tqdm(loader, desc="Evaluating")):
            blur  = sample["blur"].to(device, non_blocking=True)
            sharp = sample["sharp"].to(device, non_blocking=True)
            B     = blur.size(0)
            prior = torch.zeros(B, args.prior_dim, device=device)

            if args.no_tta:
                from torch.amp import autocast
                with autocast("cuda", enabled=args.amp):
                    out = net(blur, prior)
                pred = out[2].clamp(-0.5, 0.5).float()
            else:
                pred = tta_forward(net, blur, prior, args.amp)

            # Shift to [0, 1] for metrics
            pred_01  = pred  + 0.5
            sharp_01 = sharp + 0.5

            # PSNR — pure math, instant
            mse  = ((pred_01 - sharp_01) ** 2).mean().item()
            psnr = 10 * torch.log10(torch.tensor(1.0 / mse)).item() if mse > 0 else 100.0
            psnr_list.append(psnr)

            # LPIPS — run on ~100 evenly-spaced images only
            if i % lpips_sample_every == 0:
                lpips_sample.append(lpips_fn(pred_01, sharp_01).mean().item())

            if args.save_images:
                import numpy as np, cv2
                img = (pred_01[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
                cv2.imwrite(os.path.join(args.save_dir, f"{i:04d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    mean_psnr  = sum(psnr_list)  / len(psnr_list)
    mean_lpips = sum(lpips_sample) / len(lpips_sample) if lpips_sample else float("nan")

    mode = "No-TTA" if args.no_tta else "4-way TTA"
    print(f"\n{'='*50}")
    print(f"  Model   : {args.model_name}")
    print(f"  Mode    : {mode}")
    print(f"  Images  : {len(psnr_list)}")
    print(f"  PSNR    : {mean_psnr:.4f} dB")
    print(f"  LPIPS   : {mean_lpips:.4f}  (sampled {len(lpips_sample)} images)")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
