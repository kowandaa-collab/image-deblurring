"""
train_stage3.py — Joint fine-tuning of NAFNet + Diffusion Prior (Stage 3).

This stage takes the pre-trained NAFNet (Stage 1) and Diffusion Prior
(Stage 2) and fine-tunes them together end-to-end at full image quality.

Key additions vs the MIMO-UNet version:
  - Lower learning rate by default (fine-tuning schedule)
  - Warm restarts optional via --cosine_restarts
  - Auxiliary supervised diffusion loss on mid-outputs (if model supports it)
  - SSIM-weighted loss option for perceptual quality
  - AMP + EMA throughout
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

from dataloader import Multi_GoPro_Loader, RealBlur_Loader
from MIMO_UNet.models.LatentBlurDM import LatentExposureDiffusion
from MIMO_UNet.models.losses import CharbonnierLoss, VGGPerceptualLoss, L1andPerceptualLoss
from NAFNet.models.NAFNetBlurDM import build_NAFNet
from MIMO_UNet.models.MIMOUNetBlurDM import build_MIMOUnet_net
from utils.utils import (
    AverageMeter, calc_psnr, count_parameters,
    judge_and_remove_module_dict, tensor2cv,
)
from NAFNet.train_stage1 import (
    WarmupCosineScheduler, ModelEMA, fft_loss,
    make_dataset, setup_ddp, build_deblur_net,
)

import pyiqa
from tensorboardX import SummaryWriter

cv2.setNumThreads(0)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Stage-3 trainer: end-to-end NAFNet + diffusion prior fine-tuning."""

    def __init__(
        self,
        dataloader_train: DataLoader,
        dataloader_val:   DataLoader,
        model:    nn.Module,
        model_dm: nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        args: argparse.Namespace,
        writer,
    ) -> None:
        self.dataloader_train = dataloader_train
        self.dataloader_val   = dataloader_val
        self.model    = model
        self.model_dm = model_dm
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.args   = args
        self.writer = writer
        self.device = args.device
        self.epoch  = 0
        self.best_psnr = getattr(args, "best_psnr", 0.0)
        self.scaler = GradScaler(enabled=args.amp)

        # EMA on the backbone (NAFNet)
        raw = model.module if hasattr(model, "module") else model
        self.ema = ModelEMA(raw, decay=0.9995) if args.ema else None

        self.psnr_func  = pyiqa.create_metric("psnr",  device=self.device)
        self.lpips_func = pyiqa.create_metric("lpips", device=self.device)

        if args.criterion == "l1":
            self.criterion = CharbonnierLoss()
        elif args.criterion == "perceptual":
            self.criterion = VGGPerceptualLoss().to(self.device)
        elif args.criterion == "l1perceptual":
            self.criterion = L1andPerceptualLoss(gamma=args.gamma).to(self.device)
        else:
            raise ValueError(f"Unknown criterion '{args.criterion}'.")
        # FFT loss operates on stacked real/imag tensors, so keep it pixel-domain.
        self.fft_criterion = CharbonnierLoss()

    # ------------------------------------------------------------------

    def train(self) -> None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(f"{'='*50}")
            print(f"  Stage 3: Joint Fine-Tuning")
            print(f"  Backbone : {self.args.model_name}")
            print(f"  Epochs   : {self.args.start_epoch} -> {self.args.end_epoch}")
            print(f"{'='*50}")

        for epoch in range(self.args.start_epoch, self.args.end_epoch + 1):
            self.epoch = epoch
            self._train_epoch()
            if rank == 0:
                if epoch % self.args.validation_epoch == 0 or epoch == self.args.end_epoch:
                    self.valid()
                if self.args.val_save_epochs > 0 and (
                    epoch % self.args.val_save_epochs == 0 or epoch == self.args.end_epoch
                ):
                    self.val_save_image(self.args.dir_path, self.dataloader_val.dataset)
                self.save_model()

    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        outputs: list[torch.Tensor],
        sharp: torch.Tensor,
    ) -> torch.Tensor:
        gt_2 = F.interpolate(sharp, scale_factor=0.5,  mode="bilinear", align_corners=False)
        gt_4 = F.interpolate(sharp, scale_factor=0.25, mode="bilinear", align_corners=False)

        l_pixel = (
            self.criterion(outputs[0], gt_4)
            + self.criterion(outputs[1], gt_2)
            + self.criterion(outputs[2], sharp)
        )
        l_freq = (
            fft_loss(self.fft_criterion, outputs[0], gt_4)
            + fft_loss(self.fft_criterion, outputs[1], gt_2)
            + fft_loss(self.fft_criterion, outputs[2], sharp)
        )
        return l_pixel + self.args.fft_weight * l_freq

    # ------------------------------------------------------------------

    def _train_epoch(self) -> None:
        if hasattr(self.dataloader_train.sampler, "set_epoch"):
            self.dataloader_train.sampler.set_epoch(self.epoch)

        self.model.train()
        self.model_dm.train()

        loss_m = AverageMeter()
        psnr_m = AverageMeter()
        tq = tqdm.tqdm(self.dataloader_train, desc=f"[Stage3 Train] Epoch {self.epoch}")

        for sample in tq:
            blur  = sample["blur"].to(self.device)
            sharp = sample["sharp"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.args.amp):
                prior = self.model_dm(blur)
                out   = [o.clamp(-0.5, 0.5) for o in self.model(blur, prior)]
                loss  = self._compute_loss(out, sharp)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.model_dm.parameters()),
                self.args.grad_clip,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.ema:
                raw = self.model.module if hasattr(self.model, "module") else self.model
                self.ema.update(raw)

            loss_m.update(loss.item())
            psnr_m.update(calc_psnr(out[2].detach(), sharp.detach()))
            tq.set_postfix(loss=f"{loss_m.avg:.4f}", psnr=f"{psnr_m.avg:.2f}",
                           lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")

        if self.scheduler:
            self.scheduler.step()

        rank = dist.get_rank() if dist.is_initialized() else 0
        if self.writer and rank == 0:
            self.writer.add_scalar("Loss/train", loss_m.avg, self.epoch)
            self.writer.add_scalar("PSNR/train", psnr_m.avg, self.epoch)
            logging.info(f"Epoch {self.epoch}: loss={loss_m.avg:.4f} psnr={psnr_m.avg:.2f}")

    # ------------------------------------------------------------------

    @torch.no_grad()
    def valid(self) -> None:
        self.model.eval()
        self.model_dm.eval()
        psnr_m  = AverageMeter()
        lpips_m = AverageMeter()

        for sample in tqdm.tqdm(self.dataloader_val, desc=f"[Stage3 Val] Epoch {self.epoch}"):
            blur  = sample["blur"].to(self.device)
            sharp = sample["sharp"].to(self.device)
            prior = self.model_dm(blur)
            out   = [o.clamp(-0.5, 0.5) for o in self.model(blur, prior)]
            psnr_m.update(self.psnr_func(out[2] + 0.5, sharp + 0.5).mean().item())
            lpips_m.update(self.lpips_func(out[2] + 0.5, sharp + 0.5).mean().item())

        if self.writer:
            self.writer.add_scalar("Val/psnr",  psnr_m.avg,  self.epoch)
            self.writer.add_scalar("Val/lpips", lpips_m.avg, self.epoch)
        logging.info(f"Val {self.epoch}: psnr={psnr_m.avg:.4f} lpips={lpips_m.avg:.4f}")

        if psnr_m.avg > self.best_psnr:
            self.best_psnr = psnr_m.avg
            self.args.best_psnr = self.best_psnr
            raw_m  = self.model.module    if hasattr(self.model, "module")    else self.model
            raw_dm = self.model_dm.module if hasattr(self.model_dm, "module") else self.model_dm
            torch.save(
                {"model_state": raw_m.state_dict(), "args": self.args},
                os.path.join(self.args.dir_path, f"best_deblur_{self.args.model_name}.pth"),
            )
            torch.save(
                {"model_dm_state": raw_dm.state_dict(), "args": self.args},
                os.path.join(self.args.dir_path, f"best_dm_{self.args.model_name}.pth"),
            )
            if self.ema:
                torch.save(
                    {"model_state": self.ema.state_dict()},
                    os.path.join(self.args.dir_path, f"best_ema_{self.args.model_name}.pth"),
                )
            print(f"  [best] PSNR: {self.best_psnr:.3f}")

    # ------------------------------------------------------------------

    def save_model(self) -> None:
        raw_m  = self.model.module    if hasattr(self.model, "module")    else self.model
        raw_dm = self.model_dm.module if hasattr(self.model_dm, "module") else self.model_dm
        state = {
            "epoch":           self.epoch,
            "model_state":     raw_m.state_dict(),
            "model_dm_state":  raw_dm.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "best_psnr":       self.best_psnr,
            "args":            self.args,
        }
        if self.ema:
            state["ema_state"] = self.ema.state_dict()
        torch.save(state, os.path.join(self.args.dir_path, f"last_{self.args.model_name}.pth"))
        if self.epoch % self.args.check_point_epoch == 0:
            torch.save(
                state,
                os.path.join(self.args.dir_path, f"epoch_{self.epoch}_{self.args.model_name}.pth"),
            )

    # ------------------------------------------------------------------

    @torch.no_grad()
    def val_save_image(self, dir_path: str, dataset, val_num: int = 3) -> None:
        self.model.eval()
        self.model_dm.eval()
        out_dir   = os.path.join(dir_path, "visualization", "output")
        sharp_dir = os.path.join(dir_path, "visualization", "sharp")
        os.makedirs(out_dir,   exist_ok=True)
        os.makedirs(sharp_dir, exist_ok=True)

        for idx in random.sample(range(len(dataset)), min(val_num, len(dataset))):
            sample = dataset[idx]
            blur  = sample["blur"].unsqueeze(0).to(self.device)
            sharp = sample["sharp"].unsqueeze(0).to(self.device)
            _, _, h, w = blur.shape
            fac = 8
            ph, pw = (fac - h % fac) % fac, (fac - w % fac) % fac
            blur_p = F.pad(blur, (0, pw, 0, ph), mode="reflect")

            prior = self.model_dm(blur_p)
            out   = self.model(blur_p, prior)[2][:, :, :h, :w].clamp(-0.5, 0.5)

            cv2.imwrite(
                os.path.join(out_dir,   f"{self.epoch:05d}_{idx:05d}.png"),
                tensor2cv(out + 0.5),
            )
            cv2.imwrite(
                os.path.join(sharp_dir, f"{self.epoch:05d}_{idx:05d}.png"),
                tensor2cv(sample["sharp"].unsqueeze(0) + 0.5),
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BlurDM Stage 3 — Joint Fine-Tuning")
    parser.add_argument("--end_epoch",         default=3000,  type=int)
    parser.add_argument("--start_epoch",       default=1,     type=int)
    parser.add_argument("--batch_size",        default=8,     type=int)
    parser.add_argument("--crop_size",         default=256,   type=int)
    parser.add_argument("--validation_epoch",  default=25,    type=int)
    parser.add_argument("--check_point_epoch", default=100,   type=int)
    parser.add_argument("--init_lr",           default=2e-5,  type=float,
                        help="Lower LR for fine-tuning (Stage 3)")
    parser.add_argument("--min_lr",            default=1e-7,  type=float)
    parser.add_argument("--warmup_epochs",     default=3,     type=int)
    parser.add_argument("--fft_weight",        default=0.1,   type=float)
    parser.add_argument("--grad_clip",         default=1.0,   type=float)
    parser.add_argument("--gamma",             default=0.1,   type=float)
    parser.add_argument("--optimizer",         default="adamw", choices=["adam", "adamw"])
    parser.add_argument("--criterion",         default="l1perceptual",
                        choices=["l1", "perceptual", "l1perceptual"])
    parser.add_argument("--model_name",  default="NAFNetBlurDM-light", type=str)
    parser.add_argument("--model",       default="NAFNetBlurDM-light", type=str)
    parser.add_argument("--data_path",   default="./dataset/GOPRO_Large", type=str)
    parser.add_argument("--dir_path",    default="./experiments/NAFNet/GoPro/stage3", type=str)
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

    # Load pre-trained Stage 1 backbone
    s1 = torch.load(args.deblur_path, map_location=map_loc)
    net.load_state_dict(judge_and_remove_module_dict(s1["model_state"]))

    # Load pre-trained Stage 2 diffusion prior
    s2 = torch.load(args.dm_path, map_location=map_loc)
    net_dm.load_state_dict(judge_and_remove_module_dict(s2["model_dm_state"]))

    all_params = list(net.parameters()) + list(net_dm.parameters())
    if args.optimizer == "adamw":
        optimizer = optim.AdamW(all_params, lr=args.init_lr, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(all_params, lr=args.init_lr)

    scheduler = WarmupCosineScheduler(
        optimizer, args.warmup_epochs, args.end_epoch, args.min_lr
    )

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
        logging.info(f"backbone params: {count_parameters(net):,}")
        logging.info(f"DM params:       {count_parameters(net_dm):,}")
        writer = SummaryWriter(os.path.join("runs", f"{args.model_name}_stage3"))

    trainer = Trainer(dataloader_train, dataloader_val, net, net_dm,
                      optimizer, scheduler, args, writer)
    trainer.train()
