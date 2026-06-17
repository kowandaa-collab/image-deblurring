"""
train_standalone.py — Standalone HybridBlurDM backbone fine-tuning.

Loads the Stage 1 checkpoint (backbone + LE joint training) and fine-tunes
ONLY the backbone with no LatentEncoder involved. The FiLM / cross-attention
prior-injection layers always receive a zero prior vector; since they were
zero-initialized in Stage 1, they start as identity transforms and can learn
a per-channel bias/scale from the training signal alone.

Loss: Charbonnier-L1 + fft_weight*FFT + wavelet_weight*Wavelet + ssim_weight*SSIM
Optimizer: AdamW, init_lr=5e-5 (higher than stage1b_l2 to make real progress)
Scheduler: cosine decay from init_lr to min_lr (default 1e-7)
EMA: decay=0.999 (forced on)
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
from tensorboardX import SummaryWriter

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from HybridBlurDM.models.HybridBlurDM import build_HybridBlurDM, FiLMInjection, CrossAttentionInjection
from MIMO_UNet.models.losses import CharbonnierLoss
from utils.utils import (
    AverageMeter, batch_psnr, count_parameters,
    judge_and_remove_module_dict, tensor2cv,
)
from NAFNet.train_stage1 import (
    WarmupCosineScheduler, ModelEMA, fft_loss,
    make_dataset, setup_ddp, build_criterion,
)
from dataloader import (
    MixUpDataset, CutMixDataset, WeightedMultiDataset, make_dataloader,
)

# Reuse WaveletLoss and SSIMLoss from HybridBlurDM train_stage1
from HybridBlurDM.train_stage1 import WaveletLoss, SSIMLoss

import pyiqa

cv2.setNumThreads(0)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


# ---------------------------------------------------------------------------
# FiLM reset — ensures zero prior = identity transform
# ---------------------------------------------------------------------------

def reset_film_to_identity(model: nn.Module) -> int:
    """Reset all FiLMInjection and CrossAttentionInjection layers to identity.

    After Stage 1 training with real LE priors, the FiLM bias terms are no
    longer (scale=1, shift=0).  With a zero prior, the FiLM applies its bias
    directly, which corrupts decoder features and causes low starting PSNR.

    Resetting to identity (weight=0, bias=[1..1, 0..0]) guarantees that
    zero_prior → scale=1, shift=0 → pass-through, so the backbone starts
    from its full Stage 1 PSNR and the FiLM layers can learn from scratch.
    """
    reset_count = 0
    for m in model.modules():
        if isinstance(m, FiLMInjection):
            C = m.film.out_features // 2
            with torch.no_grad():
                m.film.weight.zero_()
                m.film.bias[:C] = 1.0
                m.film.bias[C:] = 0.0
            reset_count += 1
        elif isinstance(m, CrossAttentionInjection):
            # Zero the output projection so cross-attn starts as identity
            with torch.no_grad():
                m.out.weight.zero_()
            reset_count += 1
    return reset_count


# ---------------------------------------------------------------------------
# Standalone Trainer (no LatentEncoder)
# ---------------------------------------------------------------------------

class StandaloneTrainer:
    """Fine-tunes the HybridBlurDM backbone without any LatentEncoder.

    A zero prior vector is passed to the model's FiLM injection layers.
    Since FiLM is zero-init (scale=1, shift=0), the backbone starts in
    a well-conditioned state loaded from the Stage 1 checkpoint.
    """

    def __init__(
        self,
        dataloader_train,
        dataloader_val,
        model: nn.Module,
        optimizer,
        scheduler,
        args,
        writer,
        prior_dim: int = 256,
    ) -> None:
        self.dataloader_train = dataloader_train
        self.dataloader_val   = dataloader_val
        self.model     = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args      = args
        self.writer    = writer
        self.device    = args.device
        self.prior_dim = prior_dim
        self.epoch     = 0
        self.best_psnr = getattr(args, "best_psnr", 0.0)
        self.scaler    = GradScaler(enabled=args.amp, init_scale=2**13)
        self.accum_steps = max(1, args.accum_steps)

        # EMA always on for this fine-tuner
        self.ema = ModelEMA(model, decay=args.ema_decay)

        # Losses
        self.char_loss    = CharbonnierLoss().to(self.device)
        self.wavelet_loss = WaveletLoss().to(self.device)
        self.ssim_loss    = SSIMLoss().to(self.device)

        # Metrics
        self.psnr_func  = pyiqa.create_metric("psnr",  device=self.device)
        self.lpips_func = pyiqa.create_metric("lpips", device=self.device)

    # ------------------------------------------------------------------

    def _zero_prior(self, B: int) -> torch.Tensor:
        return torch.zeros(B, self.prior_dim, device=self.device)

    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        outputs: list[torch.Tensor],
        sharp: torch.Tensor,
    ) -> torch.Tensor:
        pred = outputs[2]  # full scale only — fastest path, maximises PSNR at test resolution
        pixel = self.char_loss(pred, sharp)
        freq  = fft_loss(self.char_loss, pred, sharp)
        return pixel + self.args.fft_weight * freq

    # ------------------------------------------------------------------

    def _train_epoch(self) -> None:
        if hasattr(self.dataloader_train.sampler, "set_epoch"):
            self.dataloader_train.sampler.set_epoch(self.epoch)

        self.model.train()
        loss_m = AverageMeter("loss")
        psnr_m = AverageMeter("psnr")
        tq     = tqdm.tqdm(self.dataloader_train, desc=f"[SA] Ep{self.epoch}")

        self.optimizer.zero_grad(set_to_none=True)
        for step, sample in enumerate(tq):
            blur  = sample["blur"].to(self.device, non_blocking=True)
            sharp = sample["sharp"].to(self.device, non_blocking=True)
            B     = blur.size(0)

            with autocast(enabled=self.args.amp):
                prior = self._zero_prior(B)
                out   = [o.clamp(-0.5, 0.5) for o in self.model(blur, prior)]
                loss  = self._compute_loss(out, sharp) / self.accum_steps

            if not torch.isfinite(loss):
                logging.warning(
                    f"Ep{self.epoch} step {step}: non-finite loss={loss.item():.6g}, skipping"
                )
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss).backward()

            if (step + 1) % self.accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                _clip_kw = {}
                if "foreach" in inspect.signature(nn.utils.clip_grad_norm_).parameters:
                    _clip_kw["foreach"] = True
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.args.grad_clip, **_clip_kw)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.ema.update(self.model)

            loss_m.update(loss.item() * self.accum_steps)
            psnr_m.update(batch_psnr(out[2].detach(), sharp.detach()))
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

    # ------------------------------------------------------------------

    @torch.no_grad()
    def valid(self) -> None:
        raw_m = self.model.module if hasattr(self.model, "module") else self.model

        # Temporarily swap in EMA weights for validation
        live_state = {k: v.clone() for k, v in raw_m.state_dict().items()}
        raw_m.load_state_dict(judge_and_remove_module_dict(self.ema.state_dict()))
        raw_m.eval()

        psnr_m  = AverageMeter("psnr")
        lpips_m = AverageMeter("lpips")

        for sample in tqdm.tqdm(self.dataloader_val, desc=f"[Val] Ep{self.epoch}"):
            blur  = sample["blur"].to(self.device, non_blocking=True)
            sharp = sample["sharp"].to(self.device, non_blocking=True)
            B     = blur.size(0)
            prior = self._zero_prior(B)
            with autocast(enabled=self.args.amp):
                out = [o.clamp(-0.5, 0.5) for o in raw_m(blur, prior)]
            psnr_m.update( self.psnr_func( out[2] + 0.5, sharp + 0.5).mean().item())
            lpips_m.update(self.lpips_func(out[2] + 0.5, sharp + 0.5).mean().item())

        # Restore live weights
        raw_m.load_state_dict(live_state)

        if self.writer:
            self.writer.add_scalar("Val/psnr",  psnr_m.avg,  self.epoch)
            self.writer.add_scalar("Val/lpips", lpips_m.avg, self.epoch)
        logging.info(f"Val Ep{self.epoch}: psnr={psnr_m.avg:.4f} lpips={lpips_m.avg:.4f}")
        print(f"  [Val Ep{self.epoch}] PSNR={psnr_m.avg:.4f}  LPIPS={lpips_m.avg:.4f}")

        if psnr_m.avg > self.best_psnr:
            self.best_psnr = psnr_m.avg
            self.args.best_psnr = self.best_psnr
            self._save_best()
            print(f"  [best] PSNR: {self.best_psnr:.4f}")

    # ------------------------------------------------------------------

    def _save_best(self) -> None:
        torch.save(
            {"model_state": self.ema.state_dict(), "args": self.args},
            os.path.join(self.args.dir_path, f"best_{self.args.model_name}.pth"),
        )

    def save_model(self) -> None:
        raw_m = self.model.module if hasattr(self.model, "module") else self.model
        state = {
            "epoch":           self.epoch,
            "model_state":     self.ema.state_dict(),   # EMA as primary
            "live_state":      raw_m.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "best_psnr":       self.best_psnr,
            "args":            self.args,
            "ema_state":       self.ema.state_dict(),
        }
        torch.save(
            state,
            os.path.join(self.args.dir_path, f"last_{self.args.model_name}.pth"),
        )
        if self.epoch % self.args.check_point_epoch == 0:
            torch.save(
                state,
                os.path.join(
                    self.args.dir_path,
                    f"epoch_{self.epoch}_{self.args.model_name}.pth",
                ),
            )

    # ------------------------------------------------------------------

    def train(self) -> None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            eff_bs = self.args.batch_size * self.accum_steps
            print(f"\n{'='*58}")
            print(f"  Mode             : Standalone backbone fine-tune")
            print(f"  Backbone         : {self.args.model_name}")
            print(f"  AMP / EMA        : {self.args.amp} / True (forced)")
            print(f"  EMA decay        : {self.args.ema_decay}")
            print(f"  Compile          : {self.args.compile}")
            print(f"  Accum steps      : {self.accum_steps} (eff. batch {eff_bs})")
            print(f"  Epochs           : {self.args.start_epoch}->{self.args.end_epoch}")
            print(f"  init_lr          : {self.args.init_lr:.2e}")
            print(f"  min_lr           : {self.args.min_lr:.2e}")
            print(f"  fft_weight       : {self.args.fft_weight}")
            print(f"  wavelet_weight   : {self.args.wavelet_weight}")
            print(f"  ssim_weight      : {self.args.ssim_weight}")
            print(f"  Prior            : zeros (no LE)")
            print(f"{'='*58}\n")

        for epoch in range(self.args.start_epoch, self.args.end_epoch + 1):
            self.epoch = epoch
            self._train_epoch()
            if rank == 0:
                if (epoch % self.args.validation_epoch == 0
                        or epoch == self.args.end_epoch):
                    self.valid()
                self.save_model()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="HybridBlurDM standalone backbone fine-tuning (no LatentEncoder)"
    )
    p.add_argument("--end_epoch",          default=100,    type=int)
    p.add_argument("--start_epoch",        default=1,      type=int)
    p.add_argument("--batch_size",         default=8,      type=int)
    p.add_argument("--accum_steps",        default=1,      type=int)
    p.add_argument("--crop_size",          default=128,    type=int)
    p.add_argument("--validation_epoch",   default=10,     type=int)
    p.add_argument("--val_subset",         default=0,      type=int)
    p.add_argument("--check_point_epoch",  default=25,     type=int)
    p.add_argument("--init_lr",            default=5e-5,   type=float,
                   help="Initial LR — 5e-5 is higher than stage1b_l2 (5e-6) to push PSNR")
    p.add_argument("--min_lr",             default=1e-7,   type=float)
    p.add_argument("--warmup_epochs",      default=3,      type=int)
    p.add_argument("--ema_decay",          default=0.999,  type=float)
    p.add_argument("--fft_weight",         default=0.1,    type=float)
    p.add_argument("--wavelet_weight",     default=0.05,   type=float)
    p.add_argument("--ssim_weight",        default=0.05,   type=float)
    p.add_argument("--grad_clip",          default=1.0,    type=float)
    p.add_argument("--model_name",         default="HybridBlurDM-light")
    p.add_argument("--data_path",          default="D:/GOPRO_Large")
    p.add_argument("--dir_path",
                   default="./experiments/HybridBlurDM/GoPro/standalone")
    p.add_argument("--stage1_ckpt",
                   default="./experiments/HybridBlurDM/GoPro/stage1/last_HybridBlurDM-light.pth",
                   help="Path to Stage 1 checkpoint — backbone weights are loaded from here")
    p.add_argument("--resume",             default=None,
                   help="Resume from a standalone checkpoint (overrides stage1_ckpt)")
    p.add_argument("--seed",               default=2023,   type=int)
    p.add_argument("--num_workers",
                   default=0 if os.name == "nt" else 8,    type=int)
    p.add_argument("--local_rank",
                   default=int(os.getenv("LOCAL_RANK", -1)), type=int)
    p.add_argument("--val_save_epochs",    default=50,     type=int)
    # Speed flags
    p.add_argument("--amp",        action="store_true")
    p.add_argument("--compile",    action="store_true")
    p.add_argument("--grad_ckpt",  action="store_true")
    # Model flags
    p.add_argument("--drop_path_rate",     default=0.0,    type=float)
    p.add_argument("--use_cross_attn",     action="store_true")
    p.add_argument("--prior_dim",          default=256,    type=int,
                   help="Latent dim expected by the model (must match Stage 1)")
    # Data aug flags (all off by default for clean fine-tune)
    p.add_argument("--mixup_alpha",        default=0.0,    type=float)
    p.add_argument("--jpeg_aug",           action="store_true")
    p.add_argument("--noise_aug",          action="store_true")
    p.add_argument("--gamma_aug",          action="store_true")
    p.add_argument("--channel_shuffle",    action="store_true")
    p.add_argument("--cache_images",       action="store_true")

    args = p.parse_args()

    device, args.local_rank = setup_ddp(args.local_rank)
    args.device = device
    num_gpus    = max(torch.cuda.device_count(), 1)

    seed = args.seed + args.local_rank
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    # ── Build backbone only ───────────────────────────────────────────────
    net = build_HybridBlurDM(
        args.model_name,
        prior_dim=args.prior_dim,
        grad_ckpt=args.grad_ckpt,
        drop_path_rate=args.drop_path_rate,
        use_cross_attn=args.use_cross_attn,
        use_deform=True,
    ).to(device)

    if args.compile and hasattr(torch, "compile"):
        print("Compiling backbone with torch.compile(mode='reduce-overhead') ...")
        net = torch.compile(net, mode="reduce-overhead")

    # ── Optimizer + scheduler ─────────────────────────────────────────────
    _adamw_extra = {}
    if "fused" in inspect.signature(optim.AdamW.__init__).parameters:
        _adamw_extra["fused"] = torch.cuda.is_available()
    optimizer = optim.AdamW(
        net.parameters(), lr=args.init_lr, weight_decay=0.01, **_adamw_extra
    )
    scheduler = WarmupCosineScheduler(
        optimizer, args.warmup_epochs, args.end_epoch, args.min_lr
    )

    # ── Checkpoint loading ────────────────────────────────────────────────
    map_loc = {"cuda:0": f"cuda:{args.local_rank}"}
    os.makedirs(args.dir_path, exist_ok=True)

    standalone_ckpt = os.path.join(args.dir_path, f"last_{args.model_name}.pth")

    if args.resume and os.path.exists(args.resume):
        # Resume from a previous standalone run
        print(f"[standalone] Resuming from: {args.resume}")
        state = torch.load(args.resume, map_location=map_loc, weights_only=False)
        args.start_epoch = state["epoch"] + 1
        args.best_psnr   = state.get("best_psnr", 0.0)
        raw = net._orig_mod if hasattr(net, "_orig_mod") else net
        raw.load_state_dict(judge_and_remove_module_dict(state["model_state"]))
        optimizer.load_state_dict(state["optimizer_state"])
        if state.get("scheduler_state"):
            scheduler.load_state_dict(state["scheduler_state"])
        print(f"[standalone] Resumed at epoch {args.start_epoch}, best PSNR={args.best_psnr:.4f}")

    elif os.path.exists(standalone_ckpt):
        # Auto-resume from last standalone checkpoint
        print(f"[standalone] Auto-resuming from: {standalone_ckpt}")
        state = torch.load(standalone_ckpt, map_location=map_loc, weights_only=False)
        args.start_epoch = state["epoch"] + 1
        args.best_psnr   = state.get("best_psnr", 0.0)
        raw = net._orig_mod if hasattr(net, "_orig_mod") else net
        raw.load_state_dict(judge_and_remove_module_dict(state["model_state"]))
        optimizer.load_state_dict(state["optimizer_state"])
        if state.get("scheduler_state"):
            scheduler.load_state_dict(state["scheduler_state"])
        print(f"[standalone] Resumed at epoch {args.start_epoch}, best PSNR={args.best_psnr:.4f}")

    else:
        # Cold-start: load backbone weights from Stage 1 checkpoint
        s1_path = args.stage1_ckpt
        if not os.path.exists(s1_path):
            raise FileNotFoundError(
                f"Stage 1 checkpoint not found: {s1_path}\n"
                "Pass --stage1_ckpt <path> or ensure it exists."
            )
        print(f"[standalone] Loading backbone from Stage 1: {s1_path}")
        state = torch.load(s1_path, map_location=map_loc, weights_only=False)

        # model_state holds EMA backbone weights (with module. prefix from DDP)
        backbone_sd = judge_and_remove_module_dict(state["model_state"])
        raw = net._orig_mod if hasattr(net, "_orig_mod") else net
        missing, unexpected = raw.load_state_dict(backbone_sd, strict=True)
        if missing:
            print(f"  WARNING: {len(missing)} missing keys: {missing[:5]}")
        if unexpected:
            print(f"  WARNING: {len(unexpected)} unexpected keys: {unexpected[:5]}")
        s1_epoch = state.get("epoch", "?")
        s1_psnr  = state.get("best_psnr", "?")
        print(f"[standalone] Stage 1 checkpoint: epoch={s1_epoch}, best_psnr={s1_psnr}")

        # Reset FiLM layers to identity (scale=1, shift=0) so that zero prior
        # is a pass-through.  Without this, Stage 1 FiLM biases (trained with
        # real LE priors) corrupt decoder features → ~22 dB instead of ~32 dB.
        n_reset = reset_film_to_identity(raw)
        print(f"[standalone] Reset {n_reset} FiLM/CrossAttn layers to identity")
        args.best_psnr = 0.0  # reset — new training run

    if dist.is_initialized():
        net = nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])

    # ── Datasets ─────────────────────────────────────────────────────────
    ds_kw = dict(
        jpeg_aug=args.jpeg_aug,
        noise_aug=args.noise_aug,
        gamma_aug=args.gamma_aug,
        channel_shuffle=args.channel_shuffle,
        cache_images=args.cache_images,
    )
    train_set = make_dataset(args.data_path, "train", args.crop_size, **ds_kw)
    if args.mixup_alpha > 0:
        train_set = MixUpDataset(train_set, args.mixup_alpha, 0.5)

    val_set = make_dataset(args.data_path, "test", args.crop_size)

    _world = dist.get_world_size() if dist.is_initialized() else 1
    _rank  = dist.get_rank()       if dist.is_initialized() else 0
    train_sampler    = DistributedSampler(train_set, num_replicas=_world, rank=_rank)
    dataloader_train = make_dataloader(
        train_set,
        batch_size=args.batch_size // num_gpus,
        sampler=train_sampler,
        num_workers=args.num_workers,
    )
    dataloader_val = make_dataloader(
        val_set,
        batch_size=max(1, args.batch_size // num_gpus // 2),
        num_workers=args.num_workers,
    )

    # ── Logging ───────────────────────────────────────────────────────────
    writer = None
    rank   = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        logging.basicConfig(
            filename=os.path.join(args.dir_path, "train.log"),
            format="%(levelname)s:%(message)s",
            level=logging.INFO,
        )
        raw_net = net._orig_mod if hasattr(net, "_orig_mod") else net
        logging.info(f"args: {args}")
        logging.info(f"backbone params: {count_parameters(raw_net):,}")
        writer = SummaryWriter(os.path.join("runs", f"standalone_{args.model_name}"))

    # ── Train ─────────────────────────────────────────────────────────────
    StandaloneTrainer(
        dataloader_train, dataloader_val,
        net, optimizer, scheduler, args, writer,
        prior_dim=args.prior_dim,
    ).train()
