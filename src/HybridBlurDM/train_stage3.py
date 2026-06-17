"""
train_stage3.py — Joint fine-tuning of HybridBlurDM + Diffusion Prior (Stage 3).

Improvements over NAFNet Stage 3:
  1. Weighted multi-scale loss: 0.5·L(full) + 0.3·L(1/2) + 0.2·L(1/4).
     Full-scale output is what PSNR measures — upweighted accordingly.
  2. SSIM loss on full-scale output (--ssim_weight default 0.1): penalises
     structural distortions that L1 and FFT miss.
  3. FFT loss applied to full scale only at 2× weight (vs NAFNet's equal per-scale).
  4. Differential learning rates:
       FiLM inject layers   → base_lr × --film_lr_mult (default 3.0)
       Backbone CNN layers  → base_lr (1×)
       Diffusion prior      → base_lr × --dm_lr_mult   (default 0.5)
     FiLM layers are newly activated by the prior signal and need more gradient;
     the backbone is already well-trained and benefits from a conservative LR.
  5. EMA on BOTH backbone and DM (--ema flag).
"""

from __future__ import annotations

import argparse
import copy
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
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from MIMO_UNet.models.LatentBlurDM import LatentExposureDiffusion
from MIMO_UNet.models.losses import CharbonnierLoss, VGGPerceptualLoss, L1andPerceptualLoss
from HybridBlurDM.models.HybridBlurDM import build_HybridBlurDM
from utils.utils import AverageMeter, calc_psnr, count_parameters, judge_and_remove_module_dict, tensor2cv
from torch.optim.lr_scheduler import (
    LinearLR, CosineAnnealingWarmRestarts, SequentialLR,
)
from NAFNet.train_stage1 import WarmupCosineScheduler, fft_loss, make_dataset, setup_ddp

import pyiqa

cv2.setNumThreads(0)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True


# ---------------------------------------------------------------------------
# SSIM loss
# ---------------------------------------------------------------------------

def _gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sobel edge loss — directly penalises blurry high-frequency reconstruction."""
    sx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=pred.dtype, device=pred.device)
    sx = sx.view(1, 1, 3, 3).repeat(3, 1, 1, 1)   # (3,1,3,3) depthwise
    sy = sx.transpose(-1, -2)
    def _edges(t):
        return F.conv2d(t, sx, padding=1, groups=3), F.conv2d(t, sy, padding=1, groups=3)
    gx_p, gy_p = _edges(pred)
    gx_t, gy_t = _edges(target)
    return ((gx_p - gx_t).abs() + (gy_p - gy_t).abs()).mean()


def _ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - SSIM on (B,3,H,W) tensors in [-0.5, 0.5]. Uses avg_pool2d (GPU-native)."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1 = F.avg_pool2d(pred,   11, stride=1, padding=5)
    mu2 = F.avg_pool2d(target, 11, stride=1, padding=5)
    mu1_sq = mu1 * mu1; mu2_sq = mu2 * mu2; mu12 = mu1 * mu2
    s1  = F.avg_pool2d(pred   * pred,   11, stride=1, padding=5) - mu1_sq
    s2  = F.avg_pool2d(target * target, 11, stride=1, padding=5) - mu2_sq
    s12 = F.avg_pool2d(pred   * target, 11, stride=1, padding=5) - mu12
    num = (2 * mu12 + C1) * (2 * s12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
    return 1.0 - (num / den.clamp(min=1e-8)).mean()


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9995) -> None:
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
# Trainer
# ---------------------------------------------------------------------------

class HybridStage3Trainer:
    """Stage-3 joint fine-tuning: HybridBlurDM backbone + diffusion prior.

    Key improvements over NAFNet Stage 3:
      - Weighted multi-scale loss [0.5, 0.3, 0.2]
      - SSIM loss on full scale
      - FFT loss on full scale only (2× weight)
      - EMA on backbone and DM
    """

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
        self.best_psnr   = getattr(args, "best_psnr", 0.0)
        self.accum_steps = max(1, getattr(args, "accum_steps", 1))
        self.scaler = GradScaler("cuda", enabled=args.amp)

        # EMA on backbone and DM
        if args.ema:
            raw_m = model.module if hasattr(model, "module") else model
            self.ema = ModelEMA(raw_m, decay=0.9995)
        else:
            self.ema = None
        self.ema_dm = None  # DM is frozen, no EMA needed

        self.psnr_func  = pyiqa.create_metric("psnr",  device=self.device)
        self.lpips_func = pyiqa.create_metric("lpips", device=self.device)
        self.fft_criterion = CharbonnierLoss()

        if args.criterion == "l1":
            self.criterion = CharbonnierLoss()
        elif args.criterion == "l2":
            self.criterion = nn.MSELoss()
        elif args.criterion == "perceptual":
            self.criterion = VGGPerceptualLoss().to(self.device)
        elif args.criterion == "l1perceptual":
            self.criterion = L1andPerceptualLoss(gamma=args.gamma).to(self.device)
        else:
            raise ValueError(f"Unknown criterion '{args.criterion}'.")

    # ------------------------------------------------------------------

    def train(self) -> None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print("=" * 55)
            print("  Stage 3: HybridBlurDM Joint Fine-Tuning")
            print(f"  Backbone : {self.args.model_name}")
            print(f"  Epochs   : {self.args.start_epoch} -> {self.args.end_epoch}")
            print(f"  Loss     : {self.args.criterion}  "
                  f"fft={self.args.fft_weight}  ssim={self.args.ssim_weight}  "
                  f"grad={self.args.grad_weight}")
            print(f"  EMA      : {self.ema is not None}")
            print(f"  Accum    : {self.accum_steps}× "
                  f"(eff. batch {self.args.batch_size * self.accum_steps})")
            print(f"  Restarts : {getattr(self.args, 'cosine_restarts', False)}")
            print(f"  LR (base/film/dm): {self.args.init_lr:.1e} / "
                  f"{self.args.init_lr * self.args.film_lr_mult:.1e} / "
                  f"{self.args.init_lr * self.args.dm_lr_mult:.1e}")
            print("=" * 55)
        freeze_until = getattr(self.args, "freeze_backbone_epochs", 0)
        if rank == 0 and freeze_until > 0:
            print(f"  Backbone frozen for first {freeze_until} epochs (FiLM-only warmup)")
        for epoch in range(self.args.start_epoch, self.args.end_epoch + 1):
            self.epoch = epoch
            if freeze_until > 0:
                if epoch <= freeze_until:
                    self._set_backbone_grad(False)
                    if rank == 0 and epoch == self.args.start_epoch:
                        print(f"  [freeze] Backbone frozen — training FiLM only")
                elif epoch == freeze_until + 1:
                    self._set_backbone_grad(True)
                    if rank == 0:
                        print(f"  [unfreeze] Ep{epoch}: backbone unfrozen — joint training")
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
        self, outputs: list[torch.Tensor], sharp: torch.Tensor
    ) -> torch.Tensor:
        gt_2 = F.interpolate(sharp, scale_factor=0.5,  mode="bilinear", align_corners=False)
        gt_4 = F.interpolate(sharp, scale_factor=0.25, mode="bilinear", align_corners=False)

        # Weighted multi-scale pixel loss: full gets 70% — maximises PSNR at test resolution
        l_pixel = (
            0.1 * self.criterion(outputs[0], gt_4) +
            0.2 * self.criterion(outputs[1], gt_2) +
            0.7 * self.criterion(outputs[2], sharp)
        )
        # Multi-scale FFT: full at 2×, half at 1× (both contribute frequency supervision)
        l_fft = (
            2.0 * fft_loss(self.fft_criterion, outputs[2], sharp) +
            1.0 * fft_loss(self.fft_criterion, outputs[1], gt_2)
        )
        # SSIM on full scale
        l_ssim = _ssim_loss(outputs[2].float(), sharp.float())
        # Sobel gradient loss — directly penalises blurry edges
        l_grad = _gradient_loss(outputs[2].float(), sharp.float())

        return (
            l_pixel
            + self.args.fft_weight  * l_fft
            + self.args.ssim_weight * l_ssim
            + self.args.grad_weight * l_grad
        )

    # ------------------------------------------------------------------

    def _set_backbone_grad(self, requires_grad: bool) -> None:
        film_names = {n for n, _ in self.model.named_parameters() if "inject" in n}
        for n, p in self.model.named_parameters():
            if n not in film_names:
                p.requires_grad_(requires_grad)

    def _train_epoch(self) -> None:
        if hasattr(self.dataloader_train.sampler, "set_epoch"):
            self.dataloader_train.sampler.set_epoch(self.epoch)
        self.model.train()
        self.model_dm.eval()  # frozen, inference only

        loss_m = AverageMeter()
        psnr_m = AverageMeter()
        tq = tqdm.tqdm(self.dataloader_train, desc=f"[S3] Ep{self.epoch}")

        import time as _time
        self.optimizer.zero_grad(set_to_none=True)
        _t0 = _time.perf_counter()
        _diag = self.epoch == 1  # print phase timing only for first epoch
        for step, sample in enumerate(tq):
            _ta = _time.perf_counter()
            blur  = sample["blur"].to(self.device)
            sharp = sample["sharp"].to(self.device)
            _tb = _time.perf_counter()

            with torch.no_grad():
                prior = self.model_dm(blur)
                if getattr(self.args, "no_prior", False):
                    prior = torch.zeros_like(prior)
            _tc = _time.perf_counter()

            with autocast("cuda", enabled=self.args.amp):
                out   = [o.clamp(-0.5, 0.5) for o in self.model(blur, prior)]
                loss  = self._compute_loss(out, sharp) / self.accum_steps
            _td = _time.perf_counter()

            self.scaler.scale(loss).backward()
            _te = _time.perf_counter()

            if (step + 1) % self.accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.args.grad_clip,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                if self.ema:
                    raw_m = self.model.module if hasattr(self.model, "module") else self.model
                    self.ema.update(raw_m)

            loss_m.update(loss.item() * self.accum_steps)
            if step % 10 == 0:
                psnr_m.update(calc_psnr(out[2].detach(), sharp.detach()))
            _elapsed = _time.perf_counter() - _t0
            _sps = _elapsed / (step + 1)
            if _diag and step < 5:
                rank = dist.get_rank() if dist.is_initialized() else 0
                if rank == 0:
                    print(f"\n[timing step {step}] data={_tb-_ta:.2f}s  dm={_tc-_tb:.2f}s  fwd={_td-_tc:.2f}s  bwd={_te-_td:.2f}s  total={_te-_ta:.2f}s", flush=True)
            tq.set_postfix(loss=f"{loss_m.avg:.4f}", psnr=f"{psnr_m.avg:.2f}",
                           lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                           sps=f"{_sps:.1f}s")

        if self.scheduler:
            self.scheduler.step()

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            if self.writer:
                self.writer.add_scalar("Loss/train", loss_m.avg, self.epoch)
                self.writer.add_scalar("PSNR/train", psnr_m.avg, self.epoch)
            logging.info(f"Ep{self.epoch}: loss={loss_m.avg:.4f} psnr={psnr_m.avg:.2f}")

    # ------------------------------------------------------------------

    @torch.no_grad()
    def valid(self) -> None:
        self.model.eval()
        self.model_dm.eval()
        psnr_m  = AverageMeter()
        lpips_m = AverageMeter()

        # Prefer EMA weights for backbone; DM is frozen (use directly)
        eval_m  = self.ema.shadow if self.ema else (self.model.module if hasattr(self.model, "module") else self.model)
        eval_dm = self.model_dm   # already eval(), no DDP wrapper

        for sample in tqdm.tqdm(self.dataloader_val, desc=f"[S3 Val] Ep{self.epoch}"):
            blur  = sample["blur"].to(self.device)
            sharp = sample["sharp"].to(self.device)
            prior = eval_dm(blur)
            if getattr(self.args, "no_prior", False):
                prior = torch.zeros_like(prior)
            out   = [o.clamp(-0.5, 0.5) for o in eval_m(blur, prior)]
            psnr_m.update(self.psnr_func(out[2] + 0.5, sharp + 0.5).mean().item())
            lpips_m.update(self.lpips_func(out[2] + 0.5, sharp + 0.5).mean().item())

        if self.writer:
            self.writer.add_scalar("Val/psnr",  psnr_m.avg,  self.epoch)
            self.writer.add_scalar("Val/lpips", lpips_m.avg, self.epoch)
        logging.info(f"Val Ep{self.epoch}: psnr={psnr_m.avg:.4f} lpips={lpips_m.avg:.4f}")

        if psnr_m.avg > self.best_psnr:
            self.best_psnr = psnr_m.avg
            self.args.best_psnr = self.best_psnr
            # Save EMA weights as best (fall back to regular weights if no EMA)
            save_m = self.ema.state_dict() if self.ema else (self.model.module if hasattr(self.model, "module") else self.model).state_dict()
            torch.save(
                {"model_state": save_m, "args": self.args},
                os.path.join(self.args.dir_path, f"best_deblur_{self.args.model_name}.pth"),
            )
            print(f"  [best] PSNR: {self.best_psnr:.4f} dB (EMA={self.ema is not None})")

    # ------------------------------------------------------------------

    def save_model(self) -> None:
        raw_m = self.model.module if hasattr(self.model, "module") else self.model
        state  = {
            "epoch":           self.epoch,
            "model_state":     raw_m.state_dict(),
            "model_dm_state":  self.model_dm.state_dict(),
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
        eval_m  = self.ema.shadow if self.ema else (self.model.module if hasattr(self.model, "module") else self.model)
        eval_dm = self.model_dm  # frozen, already eval()
        for idx in random.sample(range(len(dataset)), min(val_num, len(dataset))):
            sample = dataset[idx]
            blur  = sample["blur"].unsqueeze(0).to(self.device)
            sharp = sample["sharp"].unsqueeze(0).to(self.device)
            _, _, h, w = blur.shape
            fac = 8
            ph, pw = (fac - h % fac) % fac, (fac - w % fac) % fac
            blur_p = F.pad(blur, (0, pw, 0, ph), mode="reflect")
            prior  = eval_dm(blur_p)
            out    = eval_m(blur_p, prior)[2][:, :, :h, :w].clamp(-0.5, 0.5)
            cv2.imwrite(os.path.join(out_dir,   f"{self.epoch:05d}_{idx:05d}.png"), tensor2cv(out + 0.5))
            cv2.imwrite(os.path.join(sharp_dir, f"{self.epoch:05d}_{idx:05d}.png"), tensor2cv(sample["sharp"].unsqueeze(0) + 0.5))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HybridBlurDM Stage 3 — joint fine-tuning with diffusion prior"
    )
    parser.add_argument("--end_epoch",         default=300,   type=int)
    parser.add_argument("--start_epoch",       default=1,     type=int)
    parser.add_argument("--batch_size",        default=8,     type=int)
    parser.add_argument("--crop_size",         default=256,   type=int)
    parser.add_argument("--validation_epoch",  default=25,    type=int)
    parser.add_argument("--check_point_epoch", default=100,   type=int)
    parser.add_argument("--init_lr",           default=2e-5,  type=float)
    parser.add_argument("--min_lr",            default=1e-7,  type=float)
    parser.add_argument("--warmup_epochs",     default=3,     type=int)
    parser.add_argument("--fft_weight",        default=0.1,   type=float)
    parser.add_argument("--ssim_weight",       default=0.1,   type=float,
                        help="SSIM loss weight on full-scale output")
    parser.add_argument("--grad_weight",       default=0.05,  type=float,
                        help="Sobel gradient loss weight — penalises blurry edge reconstruction")
    parser.add_argument("--grad_clip",         default=1.0,   type=float)
    parser.add_argument("--gamma",             default=0.1,   type=float)
    parser.add_argument("--film_lr_mult",      default=3.0,   type=float,
                        help="LR multiplier for FiLM inject layers (default 3×)")
    parser.add_argument("--dm_lr_mult",        default=0.5,   type=float,
                        help="LR multiplier for diffusion prior (default 0.5×)")
    parser.add_argument("--optimizer",         default="adamw", choices=["adam", "adamw"])
    parser.add_argument("--criterion",         default="l1perceptual",
                        choices=["l1", "l2", "perceptual", "l1perceptual"])
    parser.add_argument("--model_name",        default="HybridBlurDM-light", type=str)
    parser.add_argument("--model",             default="HybridBlurDM-light", type=str)
    parser.add_argument("--data_path",         default="D:/GOPRO_Large", type=str)
    parser.add_argument("--dir_path",
                        default="./experiments/HybridBlurDM/GoPro/stage3", type=str)
    parser.add_argument("--deblur_path", required=True, type=str,
                        help="Path to Stage 1 best_deblur_*.pth")
    parser.add_argument("--dm_path",     required=True, type=str,
                        help="Path to Stage 2 best_dm_*.pth (use EMA path if available)")
    parser.add_argument("--seed",              default=2023,  type=int)
    parser.add_argument("--val_save_epochs",   default=100,   type=int)
    parser.add_argument("--resume",            default=None,  type=str)
    parser.add_argument("--num_workers",
                        default=0 if os.name == "nt" else 8, type=int)
    parser.add_argument("--local_rank",
                        default=int(os.getenv("LOCAL_RANK", -1)), type=int)
    parser.add_argument("--accum_steps",     default=1,     type=int,
                        help="Gradient accumulation steps (effective_batch = batch × accum)")
    parser.add_argument("--freeze_backbone_epochs", default=0, type=int,
                        help="Freeze backbone for first N epochs, train FiLM only, then unfreeze")
    parser.add_argument("--no_prior", action="store_true",
                        help="Zero out DM prior before passing to backbone. FiLM layers become "
                             "learnable channel-wise affine transforms. Disables DM conditioning.")
    parser.add_argument("--cosine_restarts", action="store_true",
                        help="Use CosineAnnealingWarmRestarts instead of single cosine decay")
    parser.add_argument("--restart_period",  default=50,    type=int,
                        help="T_0 epochs for first cosine restart (doubles after each restart)")
    parser.add_argument("--amp",  action="store_true")
    parser.add_argument("--ema",  action="store_true",
                        help="EMA on backbone AND DM. Recommended — saves EMA as best checkpoint.")
    args = parser.parse_args()

    device, args.local_rank = setup_ddp(args.local_rank)
    args.device = device
    num_gpus    = max(torch.cuda.device_count(), 1)

    seed = args.seed + args.local_rank
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    net    = build_HybridBlurDM(args.model_name).to(device)
    net_dm = LatentExposureDiffusion().to(device)

    map_loc = {"cuda:0": f"cuda:{args.local_rank}"}

    s1 = torch.load(args.deblur_path, map_location=map_loc)
    net.load_state_dict(judge_and_remove_module_dict(s1["model_state"]))

    if getattr(args, "no_prior", False):
        # Stage1b FiLM layers were trained with LatentEncoder prior — reset to identity
        # so they don't corrupt the backbone when prior is zeroed out.
        n_reset = 0
        for module in net.modules():
            if hasattr(module, "film") and isinstance(module.film, nn.Linear):
                nn.init.zeros_(module.film.weight)
                C = module.film.out_features // 2
                nn.init.ones_(module.film.bias[:C])
                nn.init.zeros_(module.film.bias[C:])
                n_reset += 1
        print(f"[no_prior] Reset {n_reset} FiLM layers to identity (scale=1, shift=0).")

    s2 = torch.load(args.dm_path, map_location=map_loc)
    net_dm.load_state_dict(judge_and_remove_module_dict(s2["model_dm_state"]))

    # Differential learning rates:
    #   FiLM inject layers → film_lr_mult × base_lr  (newly activated, need more signal)
    #   DM prior           → dm_lr_mult   × base_lr  (already well-trained from Stage 2)
    #   Everything else    → base_lr
    # DM is frozen (used as inference-only prior); only backbone is updated.
    net_dm.requires_grad_(False)

    film_names  = {n for n, _ in net.named_parameters() if "inject" in n}
    film_params = [p for n, p in net.named_parameters() if n in film_names]
    base_params = [p for n, p in net.named_parameters() if n not in film_names]

    param_groups = [
        {"params": film_params, "lr": args.init_lr * args.film_lr_mult},
        {"params": base_params, "lr": args.init_lr},
    ]
    optimizer = (
        optim.AdamW(param_groups, weight_decay=1e-4)
        if args.optimizer == "adamw"
        else optim.Adam(param_groups)
    )
    if args.cosine_restarts:
        # Warm restarts: warmup linearly for warmup_epochs, then cosine with doubling period.
        # e.g. with restart_period=50 and 300 epochs: restarts at ~50, 100, 200 epochs.
        warmup_sched  = LinearLR(optimizer, start_factor=1e-2, end_factor=1.0,
                                  total_iters=args.warmup_epochs)
        restart_sched = CosineAnnealingWarmRestarts(
            optimizer, T_0=args.restart_period, T_mult=2, eta_min=args.min_lr
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup_sched, restart_sched],
            milestones=[args.warmup_epochs],
        )
    else:
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
        if state.get("scheduler_state"):
            scheduler.load_state_dict(state["scheduler_state"])
    elif args.resume:
        st = torch.load(args.resume, map_location=map_loc)
        net.load_state_dict(   judge_and_remove_module_dict(st["model_state"]))
        net_dm.load_state_dict(judge_and_remove_module_dict(st["model_dm_state"]))
        os.makedirs(args.dir_path, exist_ok=True)
    else:
        os.makedirs(args.dir_path, exist_ok=True)

    # torch.compile disabled: Windows inductor cache has a rename race condition that
    # spams the terminal even with suppress_errors=True. PyTorch 2.5+cu121 eager is
    # already faster than 1.13 was on this GPU.
    print("[Stage 3] Running eager (torch.compile disabled on Windows).")

    # Only wrap in DDP for multi-GPU runs; skip for single-process (avoids gloo overhead).
    if dist.is_initialized() and dist.get_world_size() > 1:
        net = nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])
    net_dm.eval()  # frozen inference-only prior

    from dataloader import make_dataloader
    train_set = make_dataset(args.data_path, "train", args.crop_size)
    val_set   = make_dataset(args.data_path, "test",  args.crop_size)
    _world = dist.get_world_size() if dist.is_initialized() else 1
    _rank  = dist.get_rank()       if dist.is_initialized() else 0
    train_sampler = DistributedSampler(train_set, num_replicas=_world, rank=_rank)
    dataloader_train = make_dataloader(
        train_set, batch_size=args.batch_size // num_gpus,
        sampler=train_sampler, num_workers=args.num_workers,
    )
    dataloader_val = make_dataloader(
        val_set, batch_size=max(1, args.batch_size // num_gpus // 2),
        num_workers=args.num_workers, shuffle=False,
    )

    writer = None
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        logging.basicConfig(
            filename=os.path.join(args.dir_path, "train.log"),
            format="%(levelname)s:%(message)s", level=logging.INFO,
        )
        logging.info(f"args: {args}")
        logging.info(f"backbone params : {count_parameters(net):,}")
        logging.info(f"DM params       : {count_parameters(net_dm):,}")
        logging.info(f"FiLM params     : {sum(p.numel() for p in film_params):,} "
                     f"(LR {args.init_lr * args.film_lr_mult:.1e})")
        writer = SummaryWriter(os.path.join("runs", f"{args.model_name}_stage3"))

    HybridStage3Trainer(
        dataloader_train, dataloader_val,
        net, net_dm, optimizer, scheduler, args, writer,
    ).train()
