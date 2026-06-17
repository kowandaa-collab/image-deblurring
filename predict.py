"""
predict.py — Unified inference for all BlurDM backbones.

Usage:
  python predict.py --backbone NAFNet      --model_name NAFNetBlurDM-light   --model_path <s3.pth> --dm_path <s2.pth>
  python predict.py --backbone Restormer   --model_name RestormerBlurDM-light --model_path <s3.pth> --dm_path <s2.pth>
  python predict.py --backbone MIMO_UNet   --model_name MIMOUNetBlurDM        --model_path <s3.pth> --dm_path <s2.pth>
  python predict.py --backbone Stripformer --model_name StripformerPrior      --model_path <s3.pth> --dm_path <s2.pth>

Optional flags: --tta  --tile 512 --overlap 32  --ema_path <ema.pth>  --dataset GoPro+HIDE
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dataloader import Test_Loader
from utils.utils import judge_and_remove_module_dict, count_parameters


# ---------------------------------------------------------------------------
# Per-backbone factory
# ---------------------------------------------------------------------------

_PAD_FACTOR = {"NAFNet": 8, "Restormer": 8, "MIMO_UNet": 8, "Stripformer": 32, "HybridBlurDM": 8}
_OUT_INDEX  = {"NAFNet": 2, "Restormer": 2, "MIMO_UNet": 2, "Stripformer": None, "HybridBlurDM": 2}


def build_backbone(backbone: str, model_name: str) -> nn.Module:
    if backbone == "NAFNet":
        from NAFNet.models.NAFNetBlurDM import build_NAFNet
        return build_NAFNet(model_name)
    if backbone == "MIMO_UNet":
        from MIMO_UNet.models.MIMOUNetBlurDM import build_MIMOUnet_net
        return build_MIMOUnet_net(model_name)
    if backbone == "Restormer":
        from Restormer.models.RestormerBlurDM import build_Restormer
        return build_Restormer(model_name)
    if backbone == "Stripformer":
        from Stripformer.models.StripformerBlurDM import get_nets
        return get_nets(model_name)
    if backbone == "HybridBlurDM":
        from HybridBlurDM.models.HybridBlurDM import build_HybridBlurDM
        return build_HybridBlurDM(model_name)
    raise ValueError(f"Unknown backbone: {backbone!r}")


def build_dm(backbone: str) -> nn.Module:
    if backbone == "Stripformer":
        from Stripformer.models.LatentBlurDM import LatentExposureDiffusion
        return LatentExposureDiffusion(total_timestamps=5)
    from MIMO_UNet.models.LatentBlurDM import LatentExposureDiffusion
    return LatentExposureDiffusion()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _pad(t: torch.Tensor, factor: int):
    _, _, h, w = t.shape
    ph = (factor - h % factor) % factor
    pw = (factor - w % factor) % factor
    return F.pad(t, (0, pw, 0, ph), mode="reflect"), h, w


def _forward(model, dm, blur_p, out_idx):
    prior = dm(blur_p)
    out   = model(blur_p, prior)
    return out[out_idx] if out_idx is not None else out


def _tta_forward(model, dm, blur_p, out_idx):
    # 4-way TTA: original, h-flip, v-flip, hv-flip — average after un-flipping
    preds = []
    for dims in [[], [3], [2], [2, 3]]:
        x = torch.flip(blur_p, dims=dims) if dims else blur_p
        p = _forward(model, dm, x, out_idx)
        preds.append(torch.flip(p, dims=dims) if dims else p)
    return torch.stack(preds).mean(0)


@torch.no_grad()
def tile_predict(model, dm, blur, tile, overlap, tta, out_idx, factor):
    _, _, H, W = blur.shape
    stride = tile - overlap
    out  = torch.zeros_like(blur)
    wmap = torch.zeros(1, 1, H, W, device=blur.device)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y2 = min(y + tile, H);  y1 = max(0, y2 - tile)
            x2 = min(x + tile, W);  x1 = max(0, x2 - tile)
            patch = blur[:, :, y1:y2, x1:x2]
            patch_p, ph, pw = _pad(patch, factor)
            pred = (_tta_forward if tta else _forward)(model, dm, patch_p, out_idx)
            pred = pred[:, :, :ph, :pw].clamp(-0.5, 0.5)
            out[:, :, y1:y2, x1:x2]  += pred
            wmap[:, :, y1:y2, x1:x2] += 1.0

    return (out / wmap.clamp(min=1.0)).clamp(-0.5, 0.5)


# ---------------------------------------------------------------------------
# Main predict loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model, dm, args, device):
    model.eval()
    dm.eval()

    factor  = _PAD_FACTOR[args.backbone]
    out_idx = _OUT_INDEX[args.backbone]
    dataset_names = ["GoPro", "HIDE"] if args.dataset == "GoPro+HIDE" else [args.dataset]

    t_total = 0.0
    n_images = 0

    for ds_name in dataset_names:
        save_dir = os.path.join(args.dir_path, ds_name)
        os.makedirs(save_dir, exist_ok=True)
        dataset = Test_Loader(data_path=os.path.join(args.data_path, ds_name),
                              crop_size=args.crop_size)
        tq = tqdm.tqdm(range(len(dataset)), desc=f"Predict {ds_name}")

        for idx in tq:
            blur = dataset[idx]["blur"].unsqueeze(0).to(device)
            t0 = time.perf_counter()

            if args.tile > 0:
                out = tile_predict(model, dm, blur, args.tile, args.overlap,
                                   args.tta, out_idx, factor)
            else:
                blur_p, h, w = _pad(blur, factor)
                fn  = _tta_forward if args.tta else _forward
                out = fn(model, dm, blur_p, out_idx)[:, :, :h, :w].clamp(-0.5, 0.5)

            t_total  += time.perf_counter() - t0
            n_images += 1

            name = dataset.get_path(idx)["blur_path"].split(os.sep)[-1]
            save_image(out.squeeze(0).cpu() + 0.5, os.path.join(save_dir, name))

    print(f"\nDone. {n_images} images | avg {t_total / max(n_images, 1) * 1000:.1f} ms/image")
    if device.type == "cuda":
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated(device) / 1024 / 1024:.0f} MB")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="BlurDM unified inference.")
    p.add_argument("--backbone",    required=True,
                   choices=["NAFNet", "MIMO_UNet", "Restormer", "Stripformer", "HybridBlurDM"])
    p.add_argument("--model_name",  default="NAFNetBlurDM-light")
    p.add_argument("--model_path",  required=True)
    p.add_argument("--dm_path",     required=True)
    p.add_argument("--ema_path",    default=None)
    p.add_argument("--data_path",   default="./dataset/test")
    p.add_argument("--dir_path",    default="./results")
    p.add_argument("--dataset",     default="GoPro",
                   choices=["GoPro+HIDE", "GoPro", "HIDE", "Realblur_J", "RealBlur_R", "RWBI"])
    p.add_argument("--crop_size",   default=None, type=int)
    p.add_argument("--tile",        default=0,    type=int,
                   help="Tile size for large-image inference (0 = disabled).")
    p.add_argument("--overlap",     default=32,   type=int)
    p.add_argument("--tta",         action="store_true",
                   help="Horizontal-flip test-time augmentation.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.dir_path, exist_ok=True)

    net = build_backbone(args.backbone, args.model_name).to(device)
    dm  = build_dm(args.backbone).to(device)

    m = torch.load(args.model_path, map_location=device)
    key = "model_state" if "model_state" in m else "model"
    net.load_state_dict(judge_and_remove_module_dict(m.get(key, m)))

    if args.ema_path and os.path.exists(args.ema_path):
        ema = torch.load(args.ema_path, map_location=device)
        net.load_state_dict(judge_and_remove_module_dict(ema["model_state"]))
        print("Using EMA weights.")

    dm_state = torch.load(args.dm_path, map_location=device)
    dm.load_state_dict(
        judge_and_remove_module_dict(dm_state.get("model_dm_state", dm_state))
    )

    print(f"Backbone : {args.backbone} / {args.model_name}  ({count_parameters(net):,} params)")
    print(f"DM params: {count_parameters(dm):,}")
    print(f"TTA: {args.tta}  |  Tile: {args.tile or 'disabled'}")

    predict(net, dm, args, device)
