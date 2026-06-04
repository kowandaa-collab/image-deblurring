"""
train_stage1.py — Joint LE + backbone training for Stripformer.

This mirrors the structure of the other models' stage1 scripts
and trains the backbone together with the latent encoder.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import math
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
from torch.utils.data.distributed import DistributedSampler

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from dataloader import (
    Multi_GoPro_Loader, RealBlur_Loader,
    MixUpDataset, CutMixDataset, WeightedMultiDataset,
    make_dataloader,
)
from Stripformer.models.StripformerBlurDM import get_nets
from Stripformer.models.LatentEncoder import LE_arch
from Stripformer.models.losses import (
    CharbonnierLoss, VGGPerceptualLoss, L1andPerceptualLoss,
)
from utils.utils import (
    AverageMeter, batch_psnr, count_parameters,
    judge_and_remove_module_dict, tensor2cv,
)

import pyiqa
from tensorboardX import SummaryWriter

cv2.setNumThreads(0)
torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        import copy
        self.decay  = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        m = model.module if hasattr(model, "module") else model
        for s, p in zip(self.shadow.parameters(), m.parameters()):
            s.data.lerp_(p.data, 1.0 - self.decay)

    def state_dict(self) -> dict:
        return self.shadow.state_dict()


# ---------------------------------------------------------------------------
# FFT loss (reuse same helper)
# ---------------------------------------------------------------------------

def fft_loss(criterion, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pf = torch.fft.rfft2(pred,   norm="backward")
    tf = torch.fft.rfft2(target, norm="backward")
    return criterion(
        torch.stack([pf.real, pf.imag], dim=-1),
        torch.stack([tf.real, tf.imag], dim=-1),
    )


# ---------------------------------------------------------------------------
# Warmup-Cosine LR scheduler
# ---------------------------------------------------------------------------

class WarmupCosineScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(
        self, optimizer, warmup_epochs: int, total_epochs: int,
        eta_min: float = 1e-6, last_epoch: int = -1,
    ) -> None:
        self.warmup  = warmup_epochs
        self.total   = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup:
            scale = (e + 1) / max(self.warmup, 1)
        else:
            progress = (e - self.warmup) / max(self.total - self.warmup, 1)
            scale = self.eta_min / self.base_lrs[0] + 0.5 * (
                1 - self.eta_min / self.base_lrs[0]
            ) * (1 + math.cos(math.pi * progress))
        return [b * scale for b in self.base_lrs]


# ---------------------------------------------------------------------------
# DDP setup
# ---------------------------------------------------------------------------

def setup_ddp(local_rank: int):
    if local_rank != -1 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl" if (dist.is_nccl_available() and os.name != "nt") else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(0)
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                init_method=f"tcp://127.0.0.1:{os.getenv('MASTER_PORT', '29500')}",
                rank=0, world_size=1,
            )
    return device, local_rank


# ---------------------------------------------------------------------------
# Dataset / model factory
# ---------------------------------------------------------------------------

def make_dataset(
    data_path: str, mode: str, crop_size: int | None,
    jpeg_aug=False, noise_aug=False, gamma_aug=False,
    channel_shuffle=False, cache_images=False,
):
    name = os.path.basename(data_path)
    kw = dict(mode=mode, crop_size=crop_size, jpeg_aug=jpeg_aug,
              noise_aug=noise_aug, gamma_aug=gamma_aug,
              channel_shuffle=channel_shuffle, cache_images=cache_images)
    if name == "GOPRO_Large":
        return Multi_GoPro_Loader(data_path, **kw)
    if name in ("Realblur_J", "Realblur_R"):
        return RealBlur_Loader(data_path, ZeroToOne=False, **kw)
    raise ValueError(f"Unrecognised dataset: '{name}'.")


def build_deblur_net(model_name: str) -> nn.Module:
    return get_nets(model_name)


def build_criterion(args) -> nn.Module:
    if args.criterion == "l1":
        return CharbonnierLoss()
    if args.criterion == "perceptual":
        return VGGPerceptualLoss()
    if args.criterion == "l1perceptual":
        return L1andPerceptualLoss(gamma=args.gamma)
    raise ValueError(f"Unknown criterion '{args.criterion}'.")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(
        self, dataloader_train, dataloader_val,
        model, model_le, optimizer, scheduler, args, writer,
    ) -> None:
        self.dataloader_train = dataloader_train
        self.dataloader_val   = dataloader_val
        self.model    = model
        self.model_le = model_le
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.args   = args
        self.writer = writer
        self.device = args.device
        self.epoch  = 0
        self.best_psnr  = getattr(args, "best_psnr", 0.0)
        self.scaler     = GradScaler(enabled=args.amp)
        self.accum_steps = max(1, args.accum_steps)

        self.ema = ModelEMA(model, 0.999) if args.ema else None

        self.psnr_func  = pyiqa.create_metric("psnr",  device=self.device)
        self.lpips_func = pyiqa.create_metric("lpips", device=self.device)
        self.criterion  = build_criterion(args).to(self.device)

    def train(self) -> None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            eff_bs = self.args.batch_size * self.accum_steps
            print(f"\n{'='*58}")
            print(f"  Backbone         : {self.args.model_name}")
            print(f"  Criterion        : {self.args.criterion}")
            print(f"  AMP / EMA        : {self.args.amp} / {self.args.ema}")
            print(f"  Compile          : {self.args.compile}")
            print(f"  Grad Checkpoint  : {self.args.grad_ckpt}")
            print(f"  Accum steps      : {self.accum_steps} (eff. batch {eff_bs})")
            print(f"  Epochs           : {self.args.start_epoch}→{self.args.end_epoch}")
            print(f"{'='*58}\n")

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

    def _compute_loss(self, output: torch.Tensor, sharp: torch.Tensor) -> torch.Tensor:
        # StripformerPrior returns a single full-resolution tensor, not a list.
        pixel = self.criterion(output, sharp)
        freq  = fft_loss(CharbonnierLoss(), output, sharp)
        return pixel + self.args.fft_weight * freq

    def _train_epoch(self) -> None:
        if hasattr(self.dataloader_train.sampler, "set_epoch"):
            self.dataloader_train.sampler.set_epoch(self.epoch)

        self.model.train()
        self.model_le.train()

        loss_m = AverageMeter("loss")
        psnr_m = AverageMeter("psnr")
        tq     = tqdm.tqdm(self.dataloader_train, desc=f"[S1] Ep{self.epoch}")

        self.optimizer.zero_grad(set_to_none=True)
        for step, sample in enumerate(tq):
            blur  = sample["blur"].to(self.device, non_blocking=True)
            sharp = sample["sharp"].to(self.device, non_blocking=True)

            with autocast(enabled=self.args.amp):
                z   = self.model_le(blur, sharp)
                out = self.model(blur, z).clamp(-0.5, 0.5)
                loss = self._compute_loss(out, sharp) / self.accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                _clip_params = list(self.model.parameters()) + list(self.model_le.parameters())
                _clip_kw = {}
                if "foreach" in inspect.signature(nn.utils.clip_grad_norm_).parameters:
                    _clip_kw["foreach"] = True
                nn.utils.clip_grad_norm_(_clip_params, self.args.grad_clip, **_clip_kw)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.ema:
                    self.ema.update(self.model)

            loss_m.update(loss.item() * self.accum_steps)
            psnr_m.update(batch_psnr(out.detach(), sharp.detach()))
            tq.set_postfix(
                loss=f"{loss_m.avg:.4f}", psnr=f"{psnr_m.avg:.2f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
            )

        if self.scheduler:
            self.scheduler.step()

        rank = dist.get_rank() if dist.is_initialized() else 0
        if self.writer and rank == 0:
            self.writer.add_scalar("Loss/train", loss_m.avg, self.epoch)
            self.writer.add_scalar("PSNR/train", psnr_m.avg, self.epoch)
            logging.info(f"Ep{self.epoch}: loss={loss_m.avg:.4f} psnr={psnr_m.avg:.2f}")

    @torch.no_grad()
    def valid(self) -> None:
        self.model.eval()
        self.model_le.eval()
        psnr_m  = AverageMeter("psnr")
        lpips_m = AverageMeter("lpips")

        val_loader = self.dataloader_val
        if self.args.val_subset > 0:
            indices = random.sample(range(len(self.dataloader_val.dataset)),
                                    min(self.args.val_subset, len(self.dataloader_val.dataset)))
            sub = torch.utils.data.Subset(self.dataloader_val.dataset, indices)
            val_loader = make_dataloader(sub, batch_size=1, num_workers=0)

        for sample in tqdm.tqdm(val_loader, desc=f"[Val] Ep{self.epoch}"):
            blur  = sample["blur"].to(self.device, non_blocking=True)
            sharp = sample["sharp"].to(self.device, non_blocking=True)
            with autocast(enabled=self.args.amp):
                z   = self.model_le(blur, sharp)
                out = self.model(blur, z).clamp(-0.5, 0.5)
            psnr_m.update( self.psnr_func( out + 0.5, sharp + 0.5).mean().item())
            lpips_m.update(self.lpips_func(out + 0.5, sharp + 0.5).mean().item())

        if self.writer:
            self.writer.add_scalar("Val/psnr",  psnr_m.avg,  self.epoch)
            self.writer.add_scalar("Val/lpips", lpips_m.avg, self.epoch)
        logging.info(f"Val Ep{self.epoch}: psnr={psnr_m.avg:.4f} lpips={lpips_m.avg:.4f}")

        if psnr_m.avg > self.best_psnr:
            self.best_psnr = psnr_m.avg
            self.args.best_psnr = self.best_psnr
            self._save_best()
            print(f"  ✓ Best PSNR: {self.best_psnr:.3f}")

    def _save_best(self) -> None:
        raw_m  = self.model.module    if hasattr(self.model, "module") else self.model
        raw_le = self.model_le.module if hasattr(self.model_le, "module") else self.model_le
        torch.save({"model_state":    raw_m.state_dict(),  "args": self.args},
                   os.path.join(self.args.dir_path, f"best_deblur_{self.args.model_name}.pth"))
        torch.save({"model_le_state": raw_le.state_dict(), "args": self.args},
                   os.path.join(self.args.dir_path, f"best_le_{self.args.model_name}.pth"))
        if self.ema:
            torch.save({"model_state": self.ema.state_dict()},
                       os.path.join(self.args.dir_path, f"best_ema_{self.args.model_name}.pth"))

    def save_model(self) -> None:
        raw_m  = self.model.module    if hasattr(self.model, "module") else self.model
        raw_le = self.model_le.module if hasattr(self.model_le, "module") else self.model_le
        state  = {
            "epoch":           self.epoch,
            "model_state":     raw_m.state_dict(),
            "model_le_state":  raw_le.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "best_psnr":       self.best_psnr,
            "args":            self.args,
        }
        if self.ema:
            state["ema_state"] = self.ema.state_dict()
        if isinstance(self.criterion, nn.Module) and any(
            p.requires_grad for p in self.criterion.parameters()
        ):
            state["criterion_state"] = self.criterion.state_dict()
        torch.save(state, os.path.join(self.args.dir_path, f"last_{self.args.model_name}.pth"))
        if self.epoch % self.args.check_point_epoch == 0:
            torch.save(state, os.path.join(
                self.args.dir_path, f"epoch_{self.epoch}_{self.args.model_name}.pth"
            ))

    @torch.no_grad()
    def val_save_image(self, dir_path: str, dataset, val_num: int = 4) -> None:
        self.model.eval()
        self.model_le.eval()
        out_dir   = os.path.join(dir_path, "visualization", "output")
        sharp_dir = os.path.join(dir_path, "visualization", "sharp")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(sharp_dir, exist_ok=True)

        for idx in random.sample(range(len(dataset)), min(val_num, len(dataset))):
            s     = dataset[idx]
            blur  = s["blur"].unsqueeze(0).to(self.device)
            sharp = s["sharp"].unsqueeze(0).to(self.device)
            _, _, h, w = blur.shape
            ph = (8 - h % 8) % 8; pw = (8 - w % 8) % 8
            bp = F.pad(blur,  (0, pw, 0, ph), mode="reflect")
            sp = F.pad(sharp, (0, pw, 0, ph), mode="reflect")
            z  = self.model_le(bp, sp)
            o  = self.model(bp, z)[:, :, :h, :w].clamp(-0.5, 0.5)
            cv2.imwrite(os.path.join(out_dir,   f"{self.epoch:05d}_{idx:05d}.png"), tensor2cv(o + 0.5))
            cv2.imwrite(os.path.join(sharp_dir, f"{self.epoch:05d}_{idx:05d}.png"), tensor2cv(sharp + 0.5))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--end_epoch",         default=3000,  type=int)
    p.add_argument("--start_epoch",       default=1,     type=int)
    p.add_argument("--batch_size",        default=16,    type=int)
    p.add_argument("--accum_steps",       default=1,     type=int,
                   help="Gradient accumulation steps (effective_batch = batch * accum)")
    p.add_argument("--crop_size",         default=256,   type=int)
    p.add_argument("--validation_epoch",  default=25,    type=int)
    p.add_argument("--val_subset",        default=0,     type=int,
                   help="Validate on N random images (0=full val set)")
    p.add_argument("--check_point_epoch", default=100,   type=int)
    p.add_argument("--init_lr",           default=1e-4,  type=float)
    p.add_argument("--min_lr",            default=1e-6,  type=float)
    p.add_argument("--warmup_epochs",     default=5,     type=int)
    p.add_argument("--fft_weight",        default=0.1,   type=float)
    p.add_argument("--grad_clip",         default=1.0,   type=float)
    p.add_argument("--gamma",             default=0.1,   type=float)
    p.add_argument("--optimizer",         default="adamw", choices=["adam","adamw"])
    p.add_argument("--criterion",         default="l1",
                   choices=["l1","perceptual","l1perceptual"])
    p.add_argument("--model_name",   default="StripformerPrior")
    p.add_argument("--model",        default="StripformerPrior")
    p.add_argument("--data_path",    default="./dataset/GOPRO_Large")
    p.add_argument("--data_path2",   default=None)
    p.add_argument("--data_weight",  default=0.7,   type=float)
    p.add_argument("--dir_path",     default="./experiments/Stripformer/GoPro/stage1")
    p.add_argument("--seed",         default=2023,  type=int)
    p.add_argument("--val_save_epochs", default=100, type=int)
    p.add_argument("--resume",       default=None)
    p.add_argument("--num_workers",  default=0 if os.name=="nt" else 8, type=int)
    p.add_argument("--local_rank",   default=int(os.getenv("LOCAL_RANK", -1)), type=int)
    p.add_argument("--amp",      action="store_true", help="Mixed precision (AMP)")
    p.add_argument("--compile",  action="store_true", help="torch.compile() both models")
    p.add_argument("--grad_ckpt",action="store_true", help="Gradient checkpointing (saves VRAM)")
    p.add_argument("--cache_images", action="store_true", help="Cache dataset in RAM")
    p.add_argument("--drop_path_rate", default=0.0, type=float)
    p.add_argument("--ema",            action="store_true")
    p.add_argument("--mixup",          action="store_true")
    p.add_argument("--cutmix",         action="store_true")
    p.add_argument("--jpeg_aug",       action="store_true")
    p.add_argument("--noise_aug",      action="store_true")
    p.add_argument("--gamma_aug",      action="store_true")
    p.add_argument("--channel_shuffle",action="store_true")
    p.add_argument("--use_perceptual", action="store_true")
    args = p.parse_args()

    device, args.local_rank = setup_ddp(args.local_rank)
    args.device = device
    num_gpus = max(torch.cuda.device_count(), 1)

    seed = args.seed + args.local_rank
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    net    = build_deblur_net(args.model_name).to(device)
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

    ds_kw  = dict(jpeg_aug=args.jpeg_aug, noise_aug=args.noise_aug,
                  gamma_aug=args.gamma_aug, channel_shuffle=args.channel_shuffle,
                  cache_images=args.cache_images)
    train_set = make_dataset(args.data_path, "train", args.crop_size, **ds_kw)
    if args.data_path2:
        train_set2 = make_dataset(args.data_path2, "train", args.crop_size, **ds_kw)
        train_set  = WeightedMultiDataset([train_set, train_set2],
                                          [args.data_weight, 1 - args.data_weight])
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
        logging.info(f"args: {args}")
        raw_net = net._orig_mod if hasattr(net, "_orig_mod") else net
        raw_le  = net_le._orig_mod if hasattr(net_le, "_orig_mod") else net_le
        logging.info(f"backbone params : {count_parameters(raw_net):,}")
        logging.info(f"encoder params  : {count_parameters(raw_le):,}")
        writer = SummaryWriter(os.path.join("runs", args.model_name))

    Trainer(dataloader_train, dataloader_val, net, net_le,
            optimizer, scheduler, args, writer).train()
