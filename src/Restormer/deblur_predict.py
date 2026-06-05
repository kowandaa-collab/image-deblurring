"""
deblur_predict.py — Inference script for RestormerBlurDM.

Features (same as NAFNet deblur_predict.py):
  - Tile-based (overlapping patch) inference for arbitrarily large images
  - Optional EMA weight loading (--ema_path)
  - Test-Time Augmentation (--tta): horizontal flip ensemble
  - Timing + peak memory report
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
from MIMO_UNet.models.LatentBlurDM import LatentExposureDiffusion
from Restormer.models.RestormerBlurDM import build_Restormer
from utils.utils import judge_and_remove_module_dict, count_parameters

# Re-use pad/forward/tile helpers from NAFNet predict
from NAFNet.deblur_predict import _pad8, _forward, _tta_forward, tile_predict


def build_deblur_net(model_name: str) -> nn.Module:
    if model_name.startswith("RestormerBlurDM"):
        return build_Restormer(model_name)
    raise ValueError(f"Unknown model: {model_name}")


@torch.no_grad()
def predict(
    model: nn.Module,
    dm:    nn.Module,
    args:  argparse.Namespace,
    device: torch.device,
) -> None:
    model.eval()
    dm.eval()

    dataset_names = ["GoPro", "HIDE"] if args.dataset == "GoPro+HIDE" else [args.dataset]

    t_total  = 0.0
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

    print(f"\nDone. {n_images} images | avg {t_total / max(n_images, 1) * 1000:.1f} ms/image")
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        print(f"Peak VRAM: {peak_mb:.0f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",   default="./dataset/test")
    parser.add_argument("--dir_path",    default="./results/Restormer/GoPro")
    parser.add_argument("--model_path",  required=True)
    parser.add_argument("--dm_path",     required=True)
    parser.add_argument("--ema_path",    default=None)
    parser.add_argument("--model_name",  default="RestormerBlurDM-light")
    parser.add_argument("--dataset",     default="GoPro",
                        choices=["GoPro+HIDE", "GoPro", "HIDE",
                                 "Realblur_J", "RealBlur_R", "RWBI"])
    parser.add_argument("--crop_size",  default=None, type=int)
    parser.add_argument("--tile",       default=0, type=int)
    parser.add_argument("--overlap",    default=32, type=int)
    parser.add_argument("--tta",        action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.dir_path, exist_ok=True)

    net = build_deblur_net(args.model_name).to(device)
    dm  = LatentExposureDiffusion().to(device)

    m_state = torch.load(args.model_path, map_location=device)
    key     = "model_state" if "model_state" in m_state else "model"
    net.load_state_dict(judge_and_remove_module_dict(m_state.get(key, m_state)))

    if args.ema_path and os.path.exists(args.ema_path):
        ema_state = torch.load(args.ema_path, map_location=device)
        net.load_state_dict(judge_and_remove_module_dict(ema_state["model_state"]))
        print("Using EMA weights")

    dm_state = torch.load(args.dm_path, map_location=device)
    dm.load_state_dict(
        judge_and_remove_module_dict(dm_state.get("model_dm_state", dm_state))
    )

    print(f"Backbone params : {count_parameters(net):,}")
    print(f"DM params       : {count_parameters(dm):,}")
    print(f"TTA             : {args.tta}")
    print(f"Tile            : {args.tile if args.tile > 0 else 'disabled'}")

    predict(net, dm, args, device)
