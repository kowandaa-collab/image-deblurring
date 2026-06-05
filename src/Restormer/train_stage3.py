"""
train_stage3.py — Joint fine-tuning of Restormer + Diffusion Prior (Stage 3).

Takes the pre-trained Restormer backbone (Stage 1) and Diffusion Prior
(Stage 2) and fine-tunes them end-to-end at full image quality.

Lower learning rate than Stage 1 (fine-tuning schedule).  All flags from
NAFNet Stage 3 are supported (--ema, --amp, --criterion, etc.).
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
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from MIMO_UNet.models.LatentBlurDM import LatentExposureDiffusion
from MIMO_UNet.models.losses import CharbonnierLoss, VGGPerceptualLoss, L1andPerceptualLoss
from Restormer.models.RestormerBlurDM import build_Restormer
from utils.utils import AverageMeter, calc_psnr, count_parameters, judge_and_remove_module_dict, tensor2cv
from NAFNet.train_stage1 import WarmupCosineScheduler, ModelEMA, fft_loss, make_dataset, setup_ddp
from NAFNet.train_stage3 import Trainer   # reuse unchanged trainer

from tensorboardX import SummaryWriter
import pyiqa

cv2.setNumThreads(0)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def build_deblur_net(model_name: str) -> nn.Module:
    if model_name.startswith("RestormerBlurDM"):
        return build_Restormer(model_name)
    raise ValueError(f"Unknown model '{model_name}' for Restormer train_stage3.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BlurDM Stage 3 — Restormer Joint Fine-Tuning")
    parser.add_argument("--end_epoch",         default=3000,  type=int)
    parser.add_argument("--start_epoch",       default=1,     type=int)
    parser.add_argument("--batch_size",        default=8,     type=int)
    parser.add_argument("--crop_size",         default=256,   type=int)
    parser.add_argument("--validation_epoch",  default=25,    type=int)
    parser.add_argument("--check_point_epoch", default=100,   type=int)
    parser.add_argument("--init_lr",           default=2e-5,  type=float)
    parser.add_argument("--min_lr",            default=1e-7,  type=float)
    parser.add_argument("--warmup_epochs",     default=3,     type=int)
    parser.add_argument("--fft_weight",        default=0.1,   type=float)
    parser.add_argument("--grad_clip",         default=1.0,   type=float)
    parser.add_argument("--gamma",             default=0.1,   type=float)
    parser.add_argument("--optimizer",         default="adamw", choices=["adam", "adamw"])
    parser.add_argument("--criterion",         default="l1perceptual",
                        choices=["l1", "perceptual", "l1perceptual"])
    parser.add_argument("--model_name",  default="RestormerBlurDM-light", type=str)
    parser.add_argument("--model",       default="RestormerBlurDM-light", type=str)
    parser.add_argument("--data_path",   default="./dataset/GOPRO_Large", type=str)
    parser.add_argument("--dir_path",    default="./experiments/Restormer/GoPro/stage3", type=str)
    parser.add_argument("--deblur_path", required=True, type=str,
                        help="Path to Stage 1 best_deblur_*.pth")
    parser.add_argument("--dm_path",     required=True, type=str,
                        help="Path to Stage 2 best_dm_*.pth")
    parser.add_argument("--seed",        default=2023, type=int)
    parser.add_argument("--val_save_epochs", default=100, type=int)
    parser.add_argument("--resume",      default=None, type=str)
    parser.add_argument("--num_workers", default=0 if os.name == "nt" else 8, type=int)
    parser.add_argument("--local_rank",  default=int(os.getenv("LOCAL_RANK", -1)), type=int)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--ema", action="store_true")
    args = parser.parse_args()

    device, args.local_rank = setup_ddp(args.local_rank)
    args.device = device
    num_gpus = max(torch.cuda.device_count(), 1)

    seed = args.seed + args.local_rank
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    net    = build_deblur_net(args.model_name).to(device)
    net_dm = LatentExposureDiffusion().to(device)

    map_loc = {"cuda:0": f"cuda:{args.local_rank}"}

    s1 = torch.load(args.deblur_path, map_location=map_loc)
    net.load_state_dict(judge_and_remove_module_dict(s1["model_state"]))

    s2 = torch.load(args.dm_path, map_location=map_loc)
    net_dm.load_state_dict(judge_and_remove_module_dict(s2["model_dm_state"]))

    all_params = list(net.parameters()) + list(net_dm.parameters())
    if args.optimizer == "adamw":
        optimizer = optim.AdamW(all_params, lr=args.init_lr, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(all_params, lr=args.init_lr)

    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.end_epoch, args.min_lr)

    ckpt = os.path.join(args.dir_path, f"last_{args.model_name}.pth")
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=map_loc)
        args.start_epoch = state["epoch"] + 1
        args.best_psnr   = state.get("best_psnr", 0.0)
        net.load_state_dict(   judge_and_remove_module_dict(state["model_state"]))
        net_dm.load_state_dict(judge_and_remove_module_dict(state["model_dm_state"]))
        optimizer.load_state_dict(state["optimizer_state"])
        if state["scheduler_state"]:
            scheduler.load_state_dict(state["scheduler_state"])
    elif args.resume:
        st = torch.load(args.resume, map_location=map_loc)
        net.load_state_dict(   judge_and_remove_module_dict(st["model_state"]))
        net_dm.load_state_dict(judge_and_remove_module_dict(st["model_dm_state"]))
        os.makedirs(args.dir_path, exist_ok=True)
    else:
        os.makedirs(args.dir_path, exist_ok=True)

    net    = nn.parallel.DistributedDataParallel(net,    device_ids=[args.local_rank])
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
        logging.info(f"backbone params: {count_parameters(net):,}")
        logging.info(f"DM params:       {count_parameters(net_dm):,}")
        writer = SummaryWriter(os.path.join("runs", f"{args.model_name}_stage3"))

    Trainer(
        dataloader_train, dataloader_val,
        net, net_dm, optimizer, scheduler, args, writer,
    ).train()
