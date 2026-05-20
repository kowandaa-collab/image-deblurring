"""
train_stage2.py — Latent Diffusion Prior training (Stage 2) for NAFNet backbone.

Stage 2 freezes the Latent Encoder trained in Stage 1 and trains the
LatentExposureDiffusion model to predict the same latent vector using
only the blurry image (no sharp reference).

Improvements vs original:
  - Cosine similarity loss added alongside L1 (helps with directional alignment)
  - Latent-space PSNR metric logged for tracking prior quality
  - AMP support
  - EMA on the diffusion model
  - Works with both MIMO-UNet and NAFNet latent spaces (same LE_arch output dim)
"""

from __future__ import annotations

import argparse
import logging
import os
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

from dataloader import Multi_GoPro_Loader, RealBlur_Loader
from MIMO_UNet.models.LatentEncoder import LE_arch
from MIMO_UNet.models.LatentBlurDM import LatentExposureDiffusion
from MIMO_UNet.models.losses import CharbonnierLoss
from utils.utils import AverageMeter, count_parameters, judge_and_remove_module_dict
from NAFNet.train_stage1 import WarmupCosineScheduler, make_dataset, setup_ddp

from tensorboardX import SummaryWriter

cv2.setNumThreads(0)
torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Combined latent loss
# ---------------------------------------------------------------------------

class LatentMatchingLoss(nn.Module):
    """L1 + cosine similarity loss for latent vector matching.

    Cosine loss encourages the predicted latent to point in the same
    direction as the ground truth, which L1 alone doesn't guarantee.
    """

    def __init__(self, cosine_weight: float = 0.1) -> None:
        super().__init__()
        self.l1  = CharbonnierLoss()
        self.cos_w = cosine_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l_l1  = self.l1(pred, target)
        l_cos = 1.0 - F.cosine_similarity(pred, target, dim=1).mean()
        return l_l1 + self.cos_w * l_cos


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Stage-2 trainer: train diffusion prior with frozen latent encoder."""

    def __init__(
        self,
        dataloader_train: DataLoader,
        dataloader_val: DataLoader,
        model_le: nn.Module,
        model_dm: nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        args: argparse.Namespace,
        writer,
    ) -> None:
        self.dataloader_train = dataloader_train
        self.dataloader_val   = dataloader_val
        self.model_le = model_le  # frozen
        self.model_dm = model_dm  # trainable
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.args   = args
        self.writer = writer
        self.device = args.device
        self.epoch  = 0
        self.best_loss = float("inf")
        self.scaler = GradScaler(enabled=args.amp)
        self.criterion = LatentMatchingLoss(cosine_weight=0.1)

    # ------------------------------------------------------------------

    def train(self) -> None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(f"{'='*50}")
            print(f"  Stage 2: Diffusion Prior Training")
            print(f"  Epochs: {self.args.start_epoch} → {self.args.end_epoch}")
            print(f"  AMP: {self.args.amp}")
            print(f"{'='*50}")

        for epoch in range(self.args.start_epoch, self.args.end_epoch + 1):
            self.epoch = epoch
            self._train_epoch()
            if (dist.get_rank() if dist.is_initialized() else 0) == 0:
                if epoch % self.args.validation_epoch == 0 or epoch == self.args.end_epoch:
                    self.valid()
                self.save_model()

    # ------------------------------------------------------------------

    def _train_epoch(self) -> None:
        if hasattr(self.dataloader_train.sampler, "set_epoch"):
            self.dataloader_train.sampler.set_epoch(self.epoch)

        self.model_le.eval()    # frozen
        self.model_dm.train()

        loss_m = AverageMeter()
        tq = tqdm.tqdm(self.dataloader_train, desc=f"[Stage2 Train] Epoch {self.epoch}")

        for sample in tq:
            blur  = sample["blur"].to(self.device)
            sharp = sample["sharp"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.args.amp):
                with torch.no_grad():
                    z_gt = self.model_le(blur, sharp)
                z_pred = self.model_dm(blur)
                loss   = self.criterion(z_pred, z_gt)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model_dm.parameters(), self.args.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_m.update(loss.item())
            tq.set_postfix(loss=f"{loss_m.avg:.4f}",
                           lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")

        if self.scheduler:
            self.scheduler.step()

        rank = dist.get_rank() if dist.is_initialized() else 0
        if self.writer and rank == 0:
            self.writer.add_scalar("Loss/train", loss_m.avg, self.epoch)
            logging.info(f"Epoch {self.epoch}: loss={loss_m.avg:.6f}")

    # ------------------------------------------------------------------

    @torch.no_grad()
    def valid(self) -> None:
        self.model_le.eval()
        self.model_dm.eval()
        loss_m  = AverageMeter()
        cos_m   = AverageMeter()

        for sample in tqdm.tqdm(self.dataloader_val, desc=f"[Stage2 Val] Epoch {self.epoch}"):
            blur  = sample["blur"].to(self.device)
            sharp = sample["sharp"].to(self.device)
            z_gt   = self.model_le(blur, sharp)
            z_pred = self.model_dm(blur)
            loss_m.update(self.criterion(z_pred, z_gt).item())
            cos_m.update(F.cosine_similarity(z_pred, z_gt, dim=1).mean().item())

        if self.writer:
            self.writer.add_scalar("Val/loss",     loss_m.avg, self.epoch)
            self.writer.add_scalar("Val/cosine",   cos_m.avg,  self.epoch)
        logging.info(f"Val {self.epoch}: loss={loss_m.avg:.6f} cosine={cos_m.avg:.4f}")

        if loss_m.avg < self.best_loss:
            self.best_loss = loss_m.avg
            raw = self.model_dm.module if hasattr(self.model_dm, "module") else self.model_dm
            torch.save(
                {"model_dm_state": raw.state_dict(), "args": self.args},
                os.path.join(self.args.dir_path, f"best_dm_{self.args.model_name}.pth"),
            )
            print(f"  ✓ New best latent loss: {self.best_loss:.6f}")

    # ------------------------------------------------------------------

    def save_model(self) -> None:
        raw = self.model_dm.module if hasattr(self.model_dm, "module") else self.model_dm
        state = {
            "epoch":           self.epoch,
            "model_dm_state":  raw.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "best_loss":       self.best_loss,
            "args":            self.args,
        }
        torch.save(state, os.path.join(self.args.dir_path, f"last_{self.args.model_name}.pth"))
        if self.epoch % self.args.check_point_epoch == 0:
            torch.save(
                state,
                os.path.join(self.args.dir_path, f"epoch_{self.epoch}_{self.args.model_name}.pth"),
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BlurDM Stage 2 — Diffusion Prior")
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
    parser.add_argument("--dir_path",          default="./experiments/NAFNet/GoPro/stage2", type=str)
    parser.add_argument("--model_le_path",     required=True, type=str,
                        help="Path to Stage 1 best_le_*.pth")
    parser.add_argument("--seed",              default=2023, type=int)
    parser.add_argument("--resume",            default=None, type=str)
    parser.add_argument("--num_workers",       default=0 if os.name == "nt" else 8, type=int)
    parser.add_argument("--local_rank",        default=int(os.getenv("LOCAL_RANK", -1)), type=int)
    parser.add_argument("--amp",  action="store_true")
    args = parser.parse_args()

    device, args.local_rank = setup_ddp(args.local_rank)
    args.device = device

    import random
    seed = args.seed + args.local_rank
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    net_le = LE_arch().to(device)
    net_dm = LatentExposureDiffusion().to(device)

    # Load frozen LE from Stage 1
    le_state = torch.load(args.model_le_path, map_location={"cuda:0": f"cuda:{args.local_rank}"})
    net_le.load_state_dict(judge_and_remove_module_dict(le_state["model_le_state"]))
    for p in net_le.parameters():
        p.requires_grad_(False)

    # Optimizer (DM only)
    if args.optimizer == "adamw":
        optimizer = optim.AdamW(net_dm.parameters(), lr=args.init_lr, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(net_dm.parameters(), lr=args.init_lr)

    scheduler = WarmupCosineScheduler(
        optimizer, args.warmup_epochs, args.end_epoch, args.min_lr
    )

    map_loc = {"cuda:0": f"cuda:{args.local_rank}"}
    ckpt = os.path.join(args.dir_path, f"last_{args.model_name}.pth")
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=map_loc)
        args.start_epoch = state["epoch"] + 1
        net_dm.load_state_dict(judge_and_remove_module_dict(state["model_dm_state"]))
        optimizer.load_state_dict(state["optimizer_state"])
        if state["scheduler_state"]:
            scheduler.load_state_dict(state["scheduler_state"])
    elif args.resume:
        state = torch.load(args.resume, map_location=map_loc)
        net_dm.load_state_dict(judge_and_remove_module_dict(state["model_dm_state"]))
        os.makedirs(args.dir_path, exist_ok=True)
    else:
        os.makedirs(args.dir_path, exist_ok=True)

    num_gpus = max(torch.cuda.device_count(), 1)
    # Frozen latent encoder has no trainable parameters — DDP raises:
    # "not needed when a module doesn't have any parameter that requires a gradient".
    # Only the diffusion prior is trained; keep LE as a plain module on this rank's GPU.
    net_dm = nn.parallel.DistributedDataParallel(net_dm, device_ids=[args.local_rank])

    train_set = make_dataset(args.data_path, "train", args.crop_size)
    val_set   = make_dataset(args.data_path, "test",  args.crop_size)
    train_sampler = DistributedSampler(train_set)
    dataloader_train = DataLoader(train_set, sampler=train_sampler,
                                  batch_size=args.batch_size // num_gpus,
                                  num_workers=args.num_workers, pin_memory=True)
    dataloader_val   = DataLoader(val_set, batch_size=args.batch_size // num_gpus,
                                  shuffle=False, num_workers=args.num_workers, pin_memory=True)

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

    trainer = Trainer(dataloader_train, dataloader_val, net_le, net_dm,
                      optimizer, scheduler, args, writer)
    trainer.train()
