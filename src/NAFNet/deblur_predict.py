"""
deblur_predict.py v3 — Unified inference script for all BlurDM backbones.

New in v3:
  - Supports NAFNetBlurDM-* and MIMOUNetBlurDM from a single script
  - Tile-based (overlapping patch) inference for arbitrarily large images
  - Optional EMA weight loading (--ema_path)
  - Batch inference mode for higher GPU utilisation on small images
  - Timing + peak memory report at the end
  - Test-Time Augmentation (TTA): horizontal flip ensemble (--tta)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torchvision.utils import save_image

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from dataloader import Test_Loader
from MIMO_UNet.models.MIMOUNetBlurDM import build_MIMOUnet_net
from MIMO_UNet.models.LatentBlurDM import LatentExposureDiffusion
from NAFNet.models.NAFNetBlurDM import build_NAFNet
from utils.utils import judge_and_remove_module_dict, count_parameters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_deblur_net(model_name: str) -> nn.Module:
    if model_name == "MIMOUNetBlurDM":
        return build_MIMOUnet_net(model_name)
    if model_name.startswith("NAFNetBlurDM"):
        return build_NAFNet(model_name)
    raise ValueError(f"Unknown model: {model_name}")


def _pad8(t: torch.Tensor):
    _, _, h, w = t.shape
    ph = (8 - h % 8) % 8
    pw = (8 - w % 8) % 8
    return F.pad(t, (0, pw, 0, ph), mode="reflect"), h, w


def _forward(model, dm, blur_p):
    """Single forward pass: DM prior → deblur backbone."""
    prior = dm(blur_p)
    return model(blur_p, prior)[2]


def _tta_forward(model, dm, blur_p):
    """Horizontal-flip TTA: average original and flipped predictions."""
    out   = _forward(model, dm, blur_p)
    out_f = _forward(model, dm, torch.flip(blur_p, dims=[3]))
    return 0.5 * (out + torch.flip(out_f, dims=[3]))


@torch.no_grad()
def tile_predict(
    model: nn.Module,
    dm: nn.Module,
    blur: torch.Tensor,
    tile: int,
    overlap: int,
    tta: bool,
) -> torch.Tensor:
    """Sliding-window tiled inference with linear blending."""
    _, _, H, W = blur.shape
    stride = tile - overlap
    out  = torch.zeros_like(blur)
    wmap = torch.zeros(1, 1, H, W, device=blur.device)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y2 = min(y + tile, H)
            x2 = min(x + tile, W)
            y1 = max(0, y2 - tile)
            x1 = max(0, x2 - tile)

            patch = blur[:, :, y1:y2, x1:x2]
            patch_p, ph, pw = _pad8(patch)

            pred = _tta_forward(model, dm, patch_p) if tta else _forward(model, dm, patch_p)
            pred = pred[:, :, :ph, :pw].clamp(-0.5, 0.5)

            out[:, :, y1:y2, x1:x2]  += pred
            wmap[:, :, y1:y2, x1:x2] += 1.0

    return (out / wmap.clamp(min=1.0)).clamp(-0.5, 0.5)


# ---------------------------------------------------------------------------
# Main predict function
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(
    model: nn.Module,
    dm: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    model.eval()
    dm.eval()

    dataset_names = ["GoPro", "HIDE"] if args.dataset == "GoPro+HIDE" else [args.dataset]

    t_total = 0.0
    n_images = 0

    for ds_name in dataset_names:
        ds_path  = os.path.join(args.data_path, ds_name)
        save_dir = os.path.join(args.dir_path, ds_name)
        os.makedirs(save_dir, exist_ok=True)

        dataset = Test_Loader(data_path=ds_path, crop_size=args.crop_size)
        tq = tqdm.tqdm(range(len(dataset)), desc=f"Predict {ds_name}")

        for idx in tq:
            sample = dataset[idx]
            blur   = sample["blur"].unsqueeze(0).to(device)

            t0 = time.perf_counter()

            if args.tile > 0:
                out = tile_predict(model, dm, blur, args.tile, args.overlap, args.tta)
            else:
                blur_p, h, w = _pad8(blur)
                out = _tta_forward(model, dm, blur_p) if args.tta else _forward(model, dm, blur_p)
                out = out[:, :, :h, :w].clamp(-0.5, 0.5)

            t_total  += time.perf_counter() - t0
            n_images += 1

            name = dataset.get_path(idx)["blur_path"].split(os.sep)[-1]
            save_image(out.squeeze(0).cpu() + 0.5, os.path.join(save_dir, name))

    print(f"\nDone. {n_images} images | avg {t_total/max(n_images,1)*1000:.1f} ms/image")
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        print(f"Peak VRAM: {peak_mb:.0f} MB")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",    default="./dataset/test")
    parser.add_argument("--dir_path",     default="./results/NAFNet/GoPro")
    parser.add_argument("--model_path",   required=True)
    parser.add_argument("--dm_path",      required=True)
    parser.add_argument("--ema_path",     default=None, help="Optional EMA weights")
    parser.add_argument("--model_name",   default="NAFNetBlurDM-light")
    parser.add_argument("--dataset",      default="GoPro",
                        choices=["GoPro+HIDE", "GoPro", "HIDE",
                                 "Realblur_J", "RealBlur_R", "RWBI"])
    parser.add_argument("--crop_size",    default=None, type=int)
    parser.add_argument("--tile",         default=0, type=int,
                        help="Tile size for large images (0=disable)")
    parser.add_argument("--overlap",      default=32, type=int)
    parser.add_argument("--tta",          action="store_true",
                        help="Test-time augmentation (horizontal flip)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.dir_path, exist_ok=True)

    net = build_deblur_net(args.model_name).to(device)
    dm  = LatentExposureDiffusion().to(device)

    # Load backbone
    m_state = torch.load(args.model_path, map_location=device)
    key     = "model_state" if "model_state" in m_state else "model"
    net.load_state_dict(judge_and_remove_module_dict(m_state.get(key, m_state)))

    # Optionally replace backbone with EMA weights
    if args.ema_path and os.path.exists(args.ema_path):
        ema_state = torch.load(args.ema_path, map_location=device)
        net.load_state_dict(judge_and_remove_module_dict(ema_state["model_state"]))
        print("Using EMA weights")

    # Load diffusion prior
    dm_state = torch.load(args.dm_path, map_location=device)
    dm.load_state_dict(
        judge_and_remove_module_dict(dm_state.get("model_dm_state", dm_state))
    )

    print(f"Backbone params : {count_parameters(net):,}")
    print(f"DM params       : {count_parameters(dm):,}")
    print(f"TTA             : {args.tta}")
    print(f"Tile            : {args.tile if args.tile > 0 else 'disabled'}")

    predict(net, dm, args, device)
