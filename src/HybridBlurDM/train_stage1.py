"""
train_stage1.py — Joint LE + HybridBlurDM backbone training (Stage 1).

Improvements over base Trainer
================================
  1. VGGFeatureLoss: VGG16 perceptual supervision on the full-scale output
     using three feature slices (relu1_2, relu2_2, relu3_3), always in float32.

  2. WaveletLoss: Haar-DWT subbands (LL/LH/HL/HH) L1 loss on full-scale output.
     Implemented with fixed depthwise F.conv2d, no external library needed.

  3. SSIMLoss: 1 - SSIM on full-scale output. Uses torchmetrics if available,
     falls back to a manual Gaussian-window SSIM (11x11, sigma=1.5).

  4. HybridTrainer:
     - Forces EMA (ema=True always).
     - Weighted multi-scale pixel loss: 0.125*L(1/4) + 0.25*L(1/2) + 0.5*L(full).
     - Full-scale auxiliary: fft_weight*FFT + perc_weight*VGG + wavelet_weight*DWT
       + ssim_weight*SSIM.
     - Enforces AdamW weight_decay=0.01 on all param groups.
     - Validation uses EMA shadow weights (better generalisation estimate).
     - best_deblur_*.pth and last_*.pth store EMA weights as primary checkpoint.
     - Progressive crop: at epoch > prog_switch_epoch, rebuilds train dataloader
       with large_crop_size (default 256) for higher-resolution fine-tuning.
     - MixUp enabled by default via --mixup_alpha 0.4.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import inspect

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

from MIMO_UNet.models.LatentEncoder import LE_arch
from MIMO_UNet.models.losses import CharbonnierLoss
from HybridBlurDM.models.HybridBlurDM import build_HybridBlurDM
from utils.utils import count_parameters, judge_and_remove_module_dict, AverageMeter, batch_psnr
from NAFNet.train_stage1 import (
    WarmupCosineScheduler, ModelEMA, Trainer,
    build_criterion, make_dataset, setup_ddp, fft_loss,
)
from dataloader import (
    MixUpDataset, CutMixDataset, WeightedMultiDataset, make_dataloader,
)

import pyiqa

# torchmetrics SSIM — optional; manual fallback is used if not installed
try:
    from torchmetrics.functional.image import (
        structural_similarity_index_measure as _ssim_fn,
    )
    _TM_SSIM = True
except ImportError:
    try:
        from torchmetrics.functional import (
            structural_similarity_index_measure as _ssim_fn,
        )
        _TM_SSIM = True
    except ImportError:
        _ssim_fn = None
        _TM_SSIM = False

cv2.setNumThreads(0)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


# ---------------------------------------------------------------------------
# VGG perceptual loss
# ---------------------------------------------------------------------------

class VGGFeatureLoss(nn.Module):
    """VGG16 perceptual loss using relu1_2, relu2_2, relu3_3 feature slices.

    Input convention: [-0.5, 0.5] as used throughout this project.
    Internally normalises to ImageNet stats before VGG forward.
    Always runs in float32 regardless of AMP context.
    """

    def __init__(self) -> None:
        super().__init__()
        import torchvision.models as models
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        except AttributeError:
            vgg = models.vgg16(pretrained=True).features  # older torchvision

        self.slice1 = nn.Sequential(*[vgg[i] for i in range(4)])    # relu1_2
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(4, 9)]) # relu2_2
        self.slice3 = nn.Sequential(*[vgg[i] for i in range(9, 16)])# relu3_3
        for p in self.parameters():
            p.requires_grad_(False)

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return ((x + 0.5) - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self._normalize(pred.float())
        t = self._normalize(target.float())
        p1 = self.slice1(p); t1 = self.slice1(t)
        p2 = self.slice2(p1); t2 = self.slice2(t1)
        p3 = self.slice3(p2); t3 = self.slice3(t2)
        return F.l1_loss(p1, t1) + F.l1_loss(p2, t2) + F.l1_loss(p3, t3)


# ---------------------------------------------------------------------------
# Wavelet loss — Haar DWT subbands (no external library)
# ---------------------------------------------------------------------------

class WaveletLoss(nn.Module):
    """Haar-DWT L1 loss on all four subbands (LL, LH, HL, HH).

    Filters are fixed 2x2 Haar kernels stored as buffers.  Applied to each
    channel independently via reshape + F.conv2d with stride=2 (one DWT level).
    Always runs in float32.

    Haar filters: lo = [1, 1]/sqrt(2),  hi = [1, -1]/sqrt(2).
    2D filters are outer products:
        LL = lo^T * lo,  LH = lo^T * hi,  HL = hi^T * lo,  HH = hi^T * hi
    """

    def __init__(self) -> None:
        super().__init__()
        s = 2.0 ** 0.5
        lo = torch.tensor([[1.0,  1.0]]) / s   # (1, 2)
        hi = torch.tensor([[1.0, -1.0]]) / s   # (1, 2)
        ll = (lo.T @ lo).view(1, 1, 2, 2)      # approximation
        lh = (lo.T @ hi).view(1, 1, 2, 2)      # vertical detail
        hl = (hi.T @ lo).view(1, 1, 2, 2)      # horizontal detail
        hh = (hi.T @ hi).view(1, 1, 2, 2)      # diagonal detail
        self.register_buffer("filters", torch.cat([ll, lh, hl, hh], dim=0))  # (4,1,2,2)

    def _dwt(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Merge batch and channel dims so each channel is processed by all 4 filters
        x2  = x.reshape(B * C, 1, H, W)
        out = F.conv2d(x2, self.filters, stride=2)   # (B*C, 4, H//2, W//2)
        return out.reshape(B, C * 4, H // 2, W // 2)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._dwt(pred.float()), self._dwt(target.float()))


# ---------------------------------------------------------------------------
# SSIM loss — torchmetrics if available, manual Gaussian-window fallback
# ---------------------------------------------------------------------------

class SSIMLoss(nn.Module):
    """1 - SSIM loss on float32 input in [-0.5, 0.5].

    Shifts input to [0, 1] before computing SSIM with data_range=1.0.
    Uses torchmetrics.functional.structural_similarity_index_measure when
    available; falls back to a manual 11x11 Gaussian-window implementation.
    """

    def __init__(self, window_size: int = 11) -> None:
        super().__init__()
        self._win_size = window_size

    def _gaussian_window(self, C: int, device, dtype) -> torch.Tensor:
        k, sigma = self._win_size, 1.5
        coords = torch.arange(k, dtype=dtype, device=device) - k // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        w = (g[:, None] * g[None, :]).view(1, 1, k, k)
        return w.expand(C, 1, k, k).contiguous()

    def _manual_ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        C, pad = x.shape[1], self._win_size // 2
        win    = self._gaussian_window(C, x.device, x.dtype)
        mu_x   = F.conv2d(x,   win, padding=pad, groups=C)
        mu_y   = F.conv2d(y,   win, padding=pad, groups=C)
        mu_xx  = F.conv2d(x*x, win, padding=pad, groups=C)
        mu_yy  = F.conv2d(y*y, win, padding=pad, groups=C)
        mu_xy  = F.conv2d(x*y, win, padding=pad, groups=C)
        sx  = mu_xx - mu_x * mu_x
        sy  = mu_yy - mu_y * mu_y
        sxy = mu_xy - mu_x * mu_y
        C1, C2 = 1e-4, 9e-4  # (0.01)**2, (0.03)**2 for data_range=1
        num = (2 * mu_x * mu_y + C1) * (2 * sxy + C2)
        den = (mu_x**2 + mu_y**2 + C1) * (sx + sy + C2)
        return (num / den).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred.float() + 0.5   # [-0.5, 0.5] -> [0, 1]
        t = target.float() + 0.5
        if _TM_SSIM:
            return 1.0 - _ssim_fn(p, t, data_range=1.0)
        return 1.0 - self._manual_ssim(p, t)


# ---------------------------------------------------------------------------
# HybridTrainer
# ---------------------------------------------------------------------------

class HybridTrainer(Trainer):
    """HybridBlurDM-specific Trainer.

    Extends base Trainer with:
    - Forced EMA (args.ema=True overridden on init)
    - Weighted multi-scale pixel + FFT-Charbonnier + VGG perceptual loss
    - EMA shadow weights used for validation
    - EMA state saved as primary checkpoint (model_state key)
    - Progressive crop: rebuilds train loader at epoch > prog_switch_epoch
    """

    def __init__(
        self,
        dataloader_train,
        dataloader_val,
        model,
        model_le,
        optimizer,
        scheduler,
        args,
        writer,
        train_dataset=None,
        num_gpus: int = 1,
        ds_kw: dict | None = None,
    ) -> None:
        args.ema = True  # force EMA regardless of CLI flag
        super().__init__(
            dataloader_train, dataloader_val,
            model, model_le, optimizer, scheduler, args, writer,
        )
        self.vgg_loss     = VGGFeatureLoss().to(self.device)
        self.wavelet_loss = WaveletLoss().to(self.device)
        self.ssim_loss    = SSIMLoss().to(self.device)
        self._char_loss   = CharbonnierLoss()

        # Enforce AdamW weight_decay=0.01 regardless of how optimizer was built
        for pg in self.optimizer.param_groups:
            pg["weight_decay"] = 0.01

        # Progressive crop state
        self._train_dataset = train_dataset
        self._num_gpus      = num_gpus
        self._ds_kw         = ds_kw or {}
        self._prog_switched = False

    # ------------------------------------------------------------------
    # Loss

    def _compute_loss(self, outputs, sharp):
        gt_2 = F.interpolate(sharp, scale_factor=0.5,  mode="bilinear", align_corners=False)
        gt_4 = F.interpolate(sharp, scale_factor=0.25, mode="bilinear", align_corners=False)

        # Weighted multi-scale pixel loss
        pixel = (0.125 * self.criterion(outputs[0], gt_4)
                 + 0.25  * self.criterion(outputs[1], gt_2)
                 + 0.5   * self.criterion(outputs[2], sharp))

        # Weighted multi-scale FFT-Charbonnier
        freq  = (0.125 * fft_loss(self._char_loss, outputs[0], gt_4)
                 + 0.25  * fft_loss(self._char_loss, outputs[1], gt_2)
                 + 0.5   * fft_loss(self._char_loss, outputs[2], sharp))

        # Full-scale-only auxiliary losses (all handle float32 cast internally)
        perc = self.vgg_loss(outputs[2], sharp)
        wav  = self.wavelet_loss(outputs[2], sharp)
        ssim = self.ssim_loss(outputs[2], sharp)

        return (pixel
                + self.args.fft_weight     * freq
                + self.args.perc_weight    * perc
                + self.args.wavelet_weight * wav
                + self.args.ssim_weight    * ssim)

    # ------------------------------------------------------------------
    # Validation with EMA weights

    @torch.no_grad()
    def valid(self) -> None:
        raw_m = self.model.module if hasattr(self.model, "module") else self.model

        # Temporarily load EMA weights into the live model
        live_state = {k: v.clone() for k, v in raw_m.state_dict().items()}
        raw_m.load_state_dict(judge_and_remove_module_dict(self.ema.state_dict()))
        super().valid()
        raw_m.load_state_dict(live_state)

    # ------------------------------------------------------------------
    # Checkpoint saving — EMA state as primary model_state

    def _save_best(self) -> None:
        raw_le = self.model_le.module if hasattr(self.model_le, "module") else self.model_le
        torch.save(
            {"model_state": self.ema.state_dict(), "args": self.args},
            os.path.join(self.args.dir_path, f"best_deblur_{self.args.model_name}.pth"),
        )
        torch.save(
            {"model_le_state": raw_le.state_dict(), "args": self.args},
            os.path.join(self.args.dir_path, f"best_le_{self.args.model_name}.pth"),
        )

    def save_model(self) -> None:
        raw_le = self.model_le.module if hasattr(self.model_le, "module") else self.model_le
        state = {
            "epoch":           self.epoch,
            "model_state":     self.ema.state_dict(),  # EMA as primary
            "model_le_state":  raw_le.state_dict(),
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
    # Training loop with progressive crop

    def train(self) -> None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            eff_bs = self.args.batch_size * self.accum_steps
            print(f"\n{'='*58}")
            print(f"  Backbone         : {self.args.model_name}")
            print(f"  Criterion        : {self.args.criterion}")
            print(f"  AMP / EMA        : {self.args.amp} / True (forced)")
            print(f"  Compile          : {self.args.compile}")
            print(f"  Grad Checkpoint  : {self.args.grad_ckpt}")
            print(f"  Accum steps      : {self.accum_steps} (eff. batch {eff_bs})")
            print(f"  Epochs           : {self.args.start_epoch}->{self.args.end_epoch}")
            print(
                f"  Prog crop switch : epoch {self.args.prog_switch_epoch} "
                f"({self.args.crop_size}->{self.args.large_crop_size}px)"
            )
            print(f"  fft_weight       : {self.args.fft_weight}")
            print(f"  perc_weight      : {self.args.perc_weight}")
            print(f"  wavelet_weight   : {self.args.wavelet_weight}")
            print(f"  ssim_weight      : {self.args.ssim_weight}")
            print(f"  AdamW wd         : 0.01 (enforced)")
            print(f"  SSIM backend     : {'torchmetrics' if _TM_SSIM else 'manual'}")
            print(f"{'='*58}\n")

        for epoch in range(self.args.start_epoch, self.args.end_epoch + 1):
            self.epoch = epoch

            # Progressive crop: rebuild train loader once after switch epoch
            if (not self._prog_switched
                    and self._train_dataset is not None
                    and epoch > self.args.prog_switch_epoch):
                self._rebuild_train_loader()

            self._train_epoch()
            if rank == 0:
                if (epoch % self.args.validation_epoch == 0
                        or epoch == self.args.end_epoch):
                    self.valid()
                if self.args.val_save_epochs > 0 and (
                    epoch % self.args.val_save_epochs == 0
                    or epoch == self.args.end_epoch
                ):
                    self.val_save_image(self.args.dir_path, self.dataloader_val.dataset)
                self.save_model()

    def _rebuild_train_loader(self) -> None:
        self._prog_switched = True
        new_set = make_dataset(
            self.args.data_path, "train", self.args.large_crop_size, **self._ds_kw
        )
        sampler = DistributedSampler(new_set) if dist.is_initialized() else None
        self.dataloader_train = make_dataloader(
            new_set,
            batch_size=self.args.batch_size // self._num_gpus,
            sampler=sampler,
            num_workers=self.args.num_workers,
        )
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(
                f"  [prog] crop_size -> {self.args.large_crop_size} "
                f"at epoch {self.epoch}"
            )
            logging.info(
                f"Progressive crop: crop_size->{self.args.large_crop_size} "
                f"at epoch {self.epoch}"
            )


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_deblur_net(
    model_name:     str,
    grad_ckpt:      bool  = False,
    drop_path_rate: float = 0.0,
    use_cross_attn: bool  = False,
) -> nn.Module:
    if model_name.startswith("HybridBlurDM"):
        return build_HybridBlurDM(
            model_name,
            grad_ckpt=grad_ckpt,
            drop_path_rate=drop_path_rate,
            use_cross_attn=use_cross_attn,
            use_deform=True,
        )
    raise ValueError(f"Unknown model '{model_name}' for HybridBlurDM train_stage1.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="HybridBlurDM Stage 1 — backbone + LE joint training"
    )
    p.add_argument("--end_epoch",          default=300,   type=int)
    p.add_argument("--start_epoch",        default=1,     type=int)
    p.add_argument("--batch_size",         default=8,     type=int)
    p.add_argument("--accum_steps",        default=1,     type=int)
    p.add_argument("--crop_size",          default=128,   type=int)
    p.add_argument("--large_crop_size",    default=256,   type=int,
                   help="Crop size after progressive switch (prog_switch_epoch)")
    p.add_argument("--prog_switch_epoch",  default=100,   type=int,
                   help="Epoch after which to rebuild loader with large_crop_size")
    p.add_argument("--validation_epoch",   default=50,    type=int)
    p.add_argument("--val_subset",         default=0,     type=int)
    p.add_argument("--check_point_epoch",  default=100,   type=int)
    p.add_argument("--init_lr",            default=3e-4,  type=float)
    p.add_argument("--min_lr",             default=1e-6,  type=float)
    p.add_argument("--warmup_epochs",      default=5,     type=int)
    p.add_argument("--fft_weight",         default=0.1,   type=float,
                   help="Weight for FFT-magnitude Charbonnier loss (0=disabled)")
    p.add_argument("--perc_weight",        default=0.1,   type=float,
                   help="Weight for VGG16 perceptual loss (0=disabled)")
    p.add_argument("--wavelet_weight",     default=0.05,  type=float,
                   help="Weight for Haar-DWT wavelet loss on full-scale output (0=disabled)")
    p.add_argument("--ssim_weight",        default=0.05,  type=float,
                   help="Weight for SSIM loss on full-scale output (0=disabled)")
    p.add_argument("--grad_clip",          default=1.0,   type=float)
    p.add_argument("--gamma",              default=0.1,   type=float)
    p.add_argument("--optimizer",          default="adamw", choices=["adam", "adamw"])
    p.add_argument("--criterion",          default="l1",
                   choices=["deblur", "l1", "l2", "perceptual", "l1perceptual"])
    p.add_argument("--model_name",         default="HybridBlurDM-light")
    p.add_argument("--model",              default="HybridBlurDM-light")
    p.add_argument("--data_path",          default="./dataset/GOPRO_Large")
    p.add_argument("--data_path2",         default=None)
    p.add_argument("--data_weight",        default=0.7,   type=float)
    p.add_argument("--dir_path",
                   default="./experiments/HybridBlurDM/GoPro/stage1")
    p.add_argument("--seed",               default=2023,  type=int)
    p.add_argument("--val_save_epochs",    default=100,   type=int)
    p.add_argument("--resume",             default=None)
    p.add_argument("--num_workers",
                   default=0 if os.name == "nt" else 8, type=int)
    p.add_argument("--local_rank",
                   default=int(os.getenv("LOCAL_RANK", -1)), type=int)
    # Speed flags
    p.add_argument("--amp",             action="store_true")
    p.add_argument("--compile",         action="store_true")
    p.add_argument("--grad_ckpt",       action="store_true")
    p.add_argument("--cache_images",    action="store_true")
    p.add_argument("--drop_path_rate",  default=0.0, type=float)
    # Quality flags
    p.add_argument("--ema",             action="store_true")  # always forced True
    p.add_argument("--mixup",           action="store_true")
    p.add_argument("--mixup_alpha",     default=0.4,   type=float,
                   help="MixUp alpha (0=disabled; 0.4 on by default)")
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
    num_gpus    = max(torch.cuda.device_count(), 1)

    seed = args.seed + args.local_rank
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    net = build_deblur_net(
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

    all_params   = list(net.parameters()) + list(net_le.parameters())
    _adamw_extra = {}
    if "fused" in inspect.signature(optim.AdamW.__init__).parameters:
        _adamw_extra["fused"] = torch.cuda.is_available()
    optimizer = (
        optim.AdamW(all_params, lr=args.init_lr, weight_decay=0.01, **_adamw_extra)
        if args.optimizer == "adamw"
        else optim.Adam(all_params, lr=args.init_lr)
    )
    scheduler = WarmupCosineScheduler(
        optimizer, args.warmup_epochs, args.end_epoch, args.min_lr
    )

    map_loc   = {"cuda:0": f"cuda:{args.local_rank}"}
    ckpt_path = os.path.join(args.dir_path, f"last_{args.model_name}.pth")

    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=map_loc)
        args.start_epoch = state["epoch"] + 1
        args.best_psnr   = state.get("best_psnr", 0.0)
        for key, model in (("model_state", net), ("model_le_state", net_le)):
            raw = model._orig_mod if hasattr(model, "_orig_mod") else model
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

    if dist.is_initialized():
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
    if args.mixup_alpha > 0 and not args.mixup:
        # mixup_alpha-based MixUp (default on, alpha=0.4); --mixup flag takes precedence
        train_set = MixUpDataset(train_set, args.mixup_alpha, 0.5)
    elif args.mixup:
        train_set = MixUpDataset(train_set, 0.4, 0.5)
    if args.cutmix:
        train_set = CutMixDataset(train_set, 1.0, 0.5)

    val_set = make_dataset(args.data_path, "test", args.crop_size)

    _world = dist.get_world_size() if dist.is_initialized() else 1
    _rank  = dist.get_rank()       if dist.is_initialized() else 0
    train_sampler    = DistributedSampler(train_set, num_replicas=_world, rank=_rank)
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

    HybridTrainer(
        dataloader_train, dataloader_val,
        net, net_le, optimizer, scheduler, args, writer,
        train_dataset=train_set,
        num_gpus=num_gpus,
        ds_kw=ds_kw,
    ).train()
