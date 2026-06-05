"""
train_stage1.py — Joint LE + Restormer backbone training (Stage 1).

Identical training pipeline to NAFNet Stage 1, but with RestormerBlurDM as
the backbone.  All speed and quality flags (--amp, --compile, --grad_ckpt,
--ema, --mixup, etc.) are supported.

Stage 1 trains the Latent Encoder (LE_arch) jointly with the backbone using
the sharp image as the condition input to LE_arch (oracle mode).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from MIMO_UNet.models.LatentEncoder import LE_arch
from Restormer.models.RestormerBlurDM import build_Restormer
from utils.utils import count_parameters, judge_and_remove_module_dict

# Re-use the full training infrastructure from NAFNet Stage 1
from NAFNet.train_stage1 import (
    WarmupCosineScheduler,
    ModelEMA,
    Trainer,
    build_criterion,
    make_dataset,
    setup_ddp,
)
from dataloader import (
    MixUpDataset, CutMixDataset, WeightedMultiDataset, make_dataloader,
)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
import logging
import random
import numpy as np
import inspect
import torch.optim as optim

import cv2
cv2.setNumThreads(0)
torch.backends.cudnn.benchmark = True


def build_deblur_net(
    model_name: str,
    grad_ckpt: bool = False,
    drop_path_rate: float = 0.0,
    use_cross_attn: bool = False,
) -> nn.Module:
    if model_name.startswith("RestormerBlurDM"):
        return build_Restormer(
            model_name,
            grad_ckpt=grad_ckpt,
            drop_path_rate=drop_path_rate,
            use_cross_attn=use_cross_attn,
        )
    raise ValueError(f"Unknown model '{model_name}' for Restormer train_stage1.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--end_epoch",          default=3000,  type=int)
    p.add_argument("--start_epoch",        default=1,     type=int)
    p.add_argument("--batch_size",         default=8,     type=int)
    p.add_argument("--accum_steps",        default=1,     type=int)
    p.add_argument("--crop_size",          default=256,   type=int)
    p.add_argument("--validation_epoch",   default=25,    type=int)
    p.add_argument("--val_subset",         default=0,     type=int)
    p.add_argument("--check_point_epoch",  default=100,   type=int)
    p.add_argument("--init_lr",            default=3e-4,  type=float)
    p.add_argument("--min_lr",             default=1e-6,  type=float)
    p.add_argument("--warmup_epochs",      default=5,     type=int)
    p.add_argument("--fft_weight",         default=0.1,   type=float)
    p.add_argument("--grad_clip",          default=1.0,   type=float)
    p.add_argument("--gamma",              default=0.1,   type=float)
    p.add_argument("--optimizer",          default="adamw", choices=["adam", "adamw"])
    p.add_argument("--criterion",          default="deblur",
                   choices=["deblur", "l1", "perceptual", "l1perceptual"])
    p.add_argument("--model_name",  default="RestormerBlurDM-light")
    p.add_argument("--model",       default="RestormerBlurDM-light")
    p.add_argument("--data_path",   default="./dataset/GOPRO_Large")
    p.add_argument("--data_path2",  default=None)
    p.add_argument("--data_weight", default=0.7, type=float)
    p.add_argument("--dir_path",    default="./experiments/Restormer/GoPro/stage1")
    p.add_argument("--seed",        default=2023, type=int)
    p.add_argument("--val_save_epochs", default=100, type=int)
    p.add_argument("--resume",      default=None)
    p.add_argument("--num_workers", default=0 if os.name == "nt" else 8, type=int)
    p.add_argument("--local_rank",  default=int(os.getenv("LOCAL_RANK", -1)), type=int)
    # Speed flags
    p.add_argument("--amp",             action="store_true")
    p.add_argument("--compile",         action="store_true")
    p.add_argument("--grad_ckpt",       action="store_true")
    p.add_argument("--cache_images",    action="store_true")
    p.add_argument("--drop_path_rate",  default=0.0, type=float)
    # Quality flags
    p.add_argument("--ema",             action="store_true")
    p.add_argument("--mixup",           action="store_true")
    p.add_argument("--cutmix",          action="store_true")
    p.add_argument("--jpeg_aug",        action="store_true")
    p.add_argument("--noise_aug",       action="store_true")
    p.add_argument("--gamma_aug",       action="store_true")
    p.add_argument("--channel_shuffle", action="store_true")
    p.add_argument("--use_perceptual",  action="store_true")
    p.add_argument("--use_cross_attn",  action="store_true")
    args = p.parse_args()

    device, args.local_rank = setup_ddp(args.local_rank)
    args.device = device
    num_gpus = max(torch.cuda.device_count(), 1)

    seed = args.seed + args.local_rank
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    net    = build_deblur_net(
        args.model_name,
        grad_ckpt=args.grad_ckpt,
        drop_path_rate=args.drop_path_rate,
        use_cross_attn=args.use_cross_attn,
    ).to(device)
    net_le = LE_arch().to(device)

    if args.compile and hasattr(torch, "compile"):
        print("Compiling models with torch.compile(mode='reduce-overhead') …")
        net    = torch.compile(net,    mode="reduce-overhead")
        net_le = torch.compile(net_le, mode="reduce-overhead")

    all_params = list(net.parameters()) + list(net_le.parameters())
    _adamw_extra = {}
    if "fused" in inspect.signature(optim.AdamW.__init__).parameters:
        _adamw_extra["fused"] = torch.cuda.is_available()
    optimizer = (
        optim.AdamW(all_params, lr=args.init_lr, weight_decay=1e-4, **_adamw_extra)
        if args.optimizer == "adamw"
        else optim.Adam(all_params, lr=args.init_lr)
    )
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.end_epoch, args.min_lr)

    map_loc   = {"cuda:0": f"cuda:{args.local_rank}"}
    ckpt_path = os.path.join(args.dir_path, f"last_{args.model_name}.pth")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=map_loc)
        args.start_epoch = state["epoch"] + 1
        args.best_psnr   = state.get("best_psnr", 0.0)
        for key in ("model_state", "model_le_state"):
            net_target = net if key == "model_state" else net_le
            raw = net_target._orig_mod if hasattr(net_target, "_orig_mod") else net_target
            raw.load_state_dict(judge_and_remove_module_dict(state[key]))
        optimizer.load_state_dict(state["optimizer_state"])
        if state.get("scheduler_state"):
            scheduler.load_state_dict(state["scheduler_state"])
    elif args.resume:
        st = torch.load(args.resume, map_location=map_loc)
        for key, model in (("model_state", net), ("model_le_state", net_le)):
            raw = model._orig_mod if hasattr(model, "_orig_mod") else model
            raw.load_state_dict(judge_and_remove_module_dict(st[key]))
        os.makedirs(args.dir_path, exist_ok=True)
    else:
        os.makedirs(args.dir_path, exist_ok=True)

    net    = nn.parallel.DistributedDataParallel(net,    device_ids=[args.local_rank])
    net_le = nn.parallel.DistributedDataParallel(net_le, device_ids=[args.local_rank])

    ds_kw = dict(
        jpeg_aug=args.jpeg_aug, noise_aug=args.noise_aug,
        gamma_aug=args.gamma_aug, channel_shuffle=args.channel_shuffle,
        cache_images=args.cache_images,
    )
    train_set = make_dataset(args.data_path, "train", args.crop_size, **ds_kw)
    if args.data_path2:
        train_set2 = make_dataset(args.data_path2, "train", args.crop_size, **ds_kw)
        train_set  = WeightedMultiDataset(
            [train_set, train_set2], [args.data_weight, 1 - args.data_weight]
        )
    if args.mixup:
        train_set = MixUpDataset(train_set, 0.4, 0.5)
    if args.cutmix:
        train_set = CutMixDataset(train_set, 1.0, 0.5)

    val_set = make_dataset(args.data_path, "test", args.crop_size)

    train_sampler    = DistributedSampler(train_set)
    dataloader_train = make_dataloader(
        train_set, batch_size=args.batch_size // num_gpus,
        sampler=train_sampler, num_workers=args.num_workers,
    )
    dataloader_val = make_dataloader(
        val_set, batch_size=max(1, args.batch_size // num_gpus // 2),
        num_workers=args.num_workers,
    )

    writer = None
    rank   = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        logging.basicConfig(
            filename=os.path.join(args.dir_path, "train.log"),
            format="%(levelname)s:%(message)s", level=logging.INFO,
        )
        raw_net = net._orig_mod if hasattr(net, "_orig_mod") else net
        raw_le  = net_le._orig_mod if hasattr(net_le, "_orig_mod") else net_le
        logging.info(f"args: {args}")
        logging.info(f"backbone params : {count_parameters(raw_net):,}")
        logging.info(f"encoder params  : {count_parameters(raw_le):,}")
        writer = SummaryWriter(os.path.join("runs", args.model_name))

    Trainer(
        dataloader_train, dataloader_val,
        net, net_le, optimizer, scheduler, args, writer,
    ).train()
