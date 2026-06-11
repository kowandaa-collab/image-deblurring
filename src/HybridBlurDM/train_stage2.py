"""
train_stage2.py — Latent Diffusion Prior training (Stage 2) for HybridBlurDM.

Stage 2 is fully backbone-agnostic: it freezes the Stage-1 LE_arch and trains
LatentExposureDiffusion to predict the latent prior from only the blurry image.
Identical to FFTformer/train_stage2.py; only the default dir_path differs.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from MIMO_UNet.models.LatentEncoder import LE_arch
from MIMO_UNet.models.LatentBlurDM import LatentExposureDiffusion
from utils.utils import count_parameters, judge_and_remove_module_dict
from NAFNet.train_stage1 import WarmupCosineScheduler, make_dataset, setup_ddp
from NAFNet.train_stage2 import Trainer, LatentMatchingLoss
from tensorboardX import SummaryWriter

cv2.setNumThreads(0)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HybridBlurDM Stage 2 — Diffusion Prior training"
    )
    parser.add_argument("--end_epoch",         default=3000, type=int)
    parser.add_argument("--start_epoch",       default=1,    type=int)
    parser.add_argument("--batch_size",        default=16,   type=int)
    parser.add_argument("--crop_size",         default=256,  type=int)
    parser.add_argument("--validation_epoch",  default=25,   type=int)
    parser.add_argument("--check_point_epoch", default=100,  type=int)
    parser.add_argument("--init_lr",           default=1e-4, type=float)
    parser.add_argument("--min_lr",            default=1e-6, type=float)
    parser.add_argument("--warmup_epochs",     default=5,    type=int)
    parser.add_argument("--grad_clip",         default=1.0,  type=float)
    parser.add_argument("--optimizer",         default="adamw", choices=["adam", "adamw"])
    parser.add_argument("--model_name",        default="BlurDM", type=str)
    parser.add_argument("--data_path",         default="./dataset/GOPRO_Large", type=str)
    parser.add_argument("--dir_path",
                        default="./experiments/HybridBlurDM/GoPro/stage2", type=str)
    parser.add_argument("--model_le_path", required=True, type=str,
                        help="Path to Stage 1 best_le_*.pth")
    parser.add_argument("--seed",              default=2023,  type=int)
    parser.add_argument("--resume",            default=None,  type=str)
    parser.add_argument("--num_workers",
                        default=0 if os.name == "nt" else 8, type=int)
    parser.add_argument("--local_rank",
                        default=int(os.getenv("LOCAL_RANK", -1)), type=int)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    device, args.local_rank = setup_ddp(args.local_rank)
    args.device = device
    num_gpus = max(torch.cuda.device_count(), 1)

    seed = args.seed + args.local_rank
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    net_le = LE_arch().to(device)
    net_dm = LatentExposureDiffusion().to(device)

    map_loc  = {"cuda:0": f"cuda:{args.local_rank}"}
    le_state = torch.load(args.model_le_path, map_location=map_loc)
    net_le.load_state_dict(judge_and_remove_module_dict(le_state["model_le_state"]))
    for p in net_le.parameters():
        p.requires_grad_(False)

    optimizer = (
        optim.AdamW(net_dm.parameters(), lr=args.init_lr, weight_decay=1e-4)
        if args.optimizer == "adamw"
        else optim.Adam(net_dm.parameters(), lr=args.init_lr)
    )
    scheduler = WarmupCosineScheduler(
        optimizer, args.warmup_epochs, args.end_epoch, args.min_lr
    )

    ckpt = os.path.join(args.dir_path, f"last_{args.model_name}.pth")
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=map_loc)
        args.start_epoch = state["epoch"] + 1
        net_dm.load_state_dict(judge_and_remove_module_dict(state["model_dm_state"]))
        optimizer.load_state_dict(state["optimizer_state"])
        if state.get("scheduler_state"):
            scheduler.load_state_dict(state["scheduler_state"])
    elif args.resume:
        state = torch.load(args.resume, map_location=map_loc)
        net_dm.load_state_dict(judge_and_remove_module_dict(state["model_dm_state"]))
        os.makedirs(args.dir_path, exist_ok=True)
    else:
        os.makedirs(args.dir_path, exist_ok=True)

    net_dm = nn.parallel.DistributedDataParallel(net_dm, device_ids=[args.local_rank])

    train_set = make_dataset(args.data_path, "train", args.crop_size)
    val_set   = make_dataset(args.data_path, "test",  args.crop_size)
    train_sampler    = DistributedSampler(train_set)
    dataloader_train = DataLoader(
        train_set, sampler=train_sampler,
        batch_size=args.batch_size // num_gpus,
        num_workers=args.num_workers, pin_memory=True,
    )
    dataloader_val = DataLoader(
        val_set, batch_size=args.batch_size // num_gpus,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    writer = None
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        logging.basicConfig(
            filename=os.path.join(args.dir_path, "train.log"),
            format="%(levelname)s:%(message)s", level=logging.INFO,
        )
        logging.info(f"args: {args}")
        logging.info(f"DM params: {count_parameters(net_dm):,}")
        writer = SummaryWriter(os.path.join("runs", args.model_name))

    Trainer(
        dataloader_train, dataloader_val,
        net_le, net_dm, optimizer, scheduler, args, writer,
    ).train()
