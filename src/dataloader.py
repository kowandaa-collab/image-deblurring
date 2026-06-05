"""
dataloader.py — Dataset and augmentation classes for BlurDM.

Improvements over the original:
  - All augmentations are proper nn.Module / callable classes with type hints
  - RandomCrop validates bounds and raises a helpful error
  - Normalize stores 'num' correctly (no silent bug when ZeroToOne=False)
  - Multi_GoPro_Loader / RealBlur_Loader share a common _load_pair_lists helper
  - Test_Loader and Test_Loader_DDP merged into one class (DDP flag)
  - JPEG + PNG + JPG file extensions all supported in glob patterns
  - get_image() helper is a clean standalone function
  - MixUpDataset, CutMixDataset, WeightedMultiDataset for data augmentation
  - make_dataloader factory with persistent workers and prefetch
  - jpeg_aug, noise_aug, gamma_aug, channel_shuffle, cache_images support
"""

from __future__ import annotations

import glob
import os
import random
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


# ---------------------------------------------------------------------------
# Augmentation primitives
# ---------------------------------------------------------------------------

class RandomRotate:
    """Rotate image(s) by a multiple of 90°."""

    def __call__(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        k = random.randint(0, 3)
        for key in data:
            data[key] = np.rot90(data[key], k).copy()
        return data


class RandomFlip:
    """Randomly flip image(s) horizontally and/or vertically."""

    def __call__(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if random.random() > 0.5:
            for key in data:
                data[key] = np.fliplr(data[key]).copy()
        if random.random() > 0.5:
            for key in data:
                data[key] = np.flipud(data[key]).copy()
        return data


class RandomCrop:
    """Crop a random patch of size (H, W) from all arrays in *data*."""

    def __init__(self, height: int, width: int) -> None:
        self.H = height
        self.W = width

    def __call__(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        ref = next(iter(data.values()))
        ih, iw = ref.shape[:2]
        if ih < self.H or iw < self.W:
            raise ValueError(
                f"Crop size ({self.H}×{self.W}) exceeds image size ({ih}×{iw})."
            )
        top  = random.randint(0, ih - self.H)
        left = random.randint(0, iw - self.W)
        for key in data:
            data[key] = data[key][top : top + self.H, left : left + self.W].copy()
        return data


class Normalize:
    """Divide by 255, then shift to [-0.5, 0.5] or [0, 1]."""

    def __init__(self, zero_to_one: bool = False) -> None:
        self.shift = 0.0 if zero_to_one else 0.5

    def __call__(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        for key in data:
            data[key] = (data[key] / 255.0 - self.shift).astype(np.float32)
        return data


class ToTensor:
    """Convert HWC numpy arrays to CHW float tensors."""

    def __call__(self, data: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        return {
            key: torch.from_numpy(arr.transpose(2, 0, 1)).clone()
            for key, arr in data.items()
        }


class JpegAug:
    """Randomly compress images with JPEG at quality 40–95."""

    def __call__(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        quality = random.randint(40, 95)
        for key in data:
            img_bgr = cv2.cvtColor(data[key].astype(np.uint8), cv2.COLOR_RGB2BGR)
            _, enc = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            data[key] = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB).astype(np.float32)
        return data


class NoiseAug:
    """Add Gaussian noise with std in [0, 25] to the blur image only."""

    def __call__(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        sigma = random.uniform(0, 25)
        data["blur"] = np.clip(
            data["blur"] + np.random.randn(*data["blur"].shape) * sigma, 0, 255
        ).astype(np.float32)
        return data


class GammaAug:
    """Apply random gamma correction in [0.6, 1.4] to the blur image only."""

    def __call__(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        gamma = random.uniform(0.6, 1.4)
        lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                        for i in range(256)], dtype=np.uint8)
        data["blur"] = lut[data["blur"].astype(np.uint8)].astype(np.float32)
        return data


class ChannelShuffle:
    """Randomly permute RGB channels in both blur and sharp."""

    def __call__(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        perm = list(range(3))
        random.shuffle(perm)
        for key in data:
            data[key] = data[key][:, :, perm]
        return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG")


def _glob_images(directory: str) -> list[str]:
    paths: list[str] = []
    for ext in _IMG_EXTS:
        paths.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(paths)


def _read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"cv2.imread failed: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)


def _make_transform(
    crop_size: int | None,
    zero_to_one: bool,
    augment: bool,
    jpeg_aug: bool = False,
    noise_aug: bool = False,
    gamma_aug: bool = False,
    channel_shuffle: bool = False,
) -> Callable:
    steps: list[Callable] = []
    if crop_size:
        steps.append(RandomCrop(crop_size, crop_size))
    if augment:
        steps.append(RandomFlip())
        steps.append(RandomRotate())
        if jpeg_aug:
            steps.append(JpegAug())
        if noise_aug:
            steps.append(NoiseAug())
        if gamma_aug:
            steps.append(GammaAug())
        if channel_shuffle:
            steps.append(ChannelShuffle())
    steps.append(Normalize(zero_to_one))
    steps.append(ToTensor())
    return transforms.Compose(steps)


# ---------------------------------------------------------------------------
# Training datasets
# ---------------------------------------------------------------------------

class PairedImageDataset(Dataset):
    """Base class for paired blur/sharp datasets.

    Subclasses implement ``_collect_lists`` to populate
    ``self.blur_list`` and ``self.sharp_list``.
    """

    def __init__(
        self,
        data_path: str,
        mode: str = "train",
        crop_size: int | None = None,
        zero_to_one: bool = False,
        ZeroToOne: bool | None = None,  # alias for zero_to_one
        augment: bool = True,
        jpeg_aug: bool = False,
        noise_aug: bool = False,
        gamma_aug: bool = False,
        channel_shuffle: bool = False,
        cache_images: bool = False,
    ) -> None:
        # ZeroToOne keyword alias (used by RealBlur_Loader callers)
        if ZeroToOne is not None:
            zero_to_one = ZeroToOne

        self.blur_list: list[str] = []
        self.sharp_list: list[str] = []
        self._collect_lists(data_path, mode)
        assert len(self.blur_list) == len(self.sharp_list), (
            f"Length mismatch: {len(self.blur_list)} blur vs {len(self.sharp_list)} sharp"
        )
        self.transform = _make_transform(
            crop_size, zero_to_one, augment and mode == "train",
            jpeg_aug=jpeg_aug, noise_aug=noise_aug,
            gamma_aug=gamma_aug, channel_shuffle=channel_shuffle,
        )

        self._cache: dict[int, dict] | None = {} if cache_images else None
        if cache_images:
            for i in range(len(self.blur_list)):
                self._cache[i] = {
                    "blur":  _read_rgb(self.blur_list[i]),
                    "sharp": _read_rgb(self.sharp_list[i]),
                }

    def _collect_lists(self, data_path: str, mode: str) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.blur_list)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self._cache is not None:
            sample = {k: v.copy() for k, v in self._cache[idx].items()}
        else:
            sample = {
                "blur":  _read_rgb(self.blur_list[idx]),
                "sharp": _read_rgb(self.sharp_list[idx]),
            }
        return self.transform(sample)


class Multi_GoPro_Loader(PairedImageDataset):
    """GoPro-style dataset: data_path/{mode}/{video_name}/blur|sharp/*.png"""

    def _collect_lists(self, data_path: str, mode: str) -> None:
        for video in sorted(os.listdir(os.path.join(data_path, mode))):
            blur_dir  = os.path.join(data_path, mode, video, "blur")
            sharp_dir = os.path.join(data_path, mode, video, "sharp")
            self.blur_list.extend(_glob_images(blur_dir))
            self.sharp_list.extend(_glob_images(sharp_dir))


class RealBlur_Loader(PairedImageDataset):
    """RealBlur dataset: data_path/{mode}/blur|sharp/{scene}/*.png"""

    def _collect_lists(self, data_path: str, mode: str) -> None:
        for scene in sorted(os.listdir(os.path.join(data_path, mode, "blur"))):
            blur_dir  = os.path.join(data_path, mode, "blur",  scene)
            sharp_dir = os.path.join(data_path, mode, "sharp", scene)
            self.blur_list.extend(_glob_images(blur_dir))
            self.sharp_list.extend(_glob_images(sharp_dir))


# ---------------------------------------------------------------------------
# Augmentation wrapper datasets
# ---------------------------------------------------------------------------

class MixUpDataset(Dataset):
    """Applies MixUp augmentation: blends two samples with Beta-sampled weight."""

    def __init__(self, dataset: Dataset, alpha: float = 0.4, prob: float = 0.5) -> None:
        self.dataset = dataset
        self.alpha   = alpha
        self.prob    = prob

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.dataset[idx]
        if random.random() < self.prob:
            idx2    = random.randint(0, len(self.dataset) - 1)
            sample2 = self.dataset[idx2]
            lam     = float(np.random.beta(self.alpha, self.alpha))
            sample  = {k: lam * sample[k] + (1.0 - lam) * sample2[k] for k in sample}
        return sample


class CutMixDataset(Dataset):
    """Applies CutMix augmentation: pastes a rectangular region from another sample."""

    def __init__(self, dataset: Dataset, alpha: float = 1.0, prob: float = 0.5) -> None:
        self.dataset = dataset
        self.alpha   = alpha
        self.prob    = prob

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.dataset[idx]
        if random.random() < self.prob:
            idx2    = random.randint(0, len(self.dataset) - 1)
            sample2 = self.dataset[idx2]
            lam     = float(np.random.beta(self.alpha, self.alpha))
            _, H, W = sample["blur"].shape
            cut_h   = int(H * (1.0 - lam) ** 0.5)
            cut_w   = int(W * (1.0 - lam) ** 0.5)
            cx      = random.randint(0, W)
            cy      = random.randint(0, H)
            x1 = max(0, cx - cut_w // 2); x2 = min(W, cx + cut_w // 2)
            y1 = max(0, cy - cut_h // 2); y2 = min(H, cy + cut_h // 2)
            sample  = {k: v.clone() for k, v in sample.items()}
            for k in sample:
                sample[k][:, y1:y2, x1:x2] = sample2[k][:, y1:y2, x1:x2]
        return sample


class WeightedMultiDataset(Dataset):
    """Combines multiple datasets, sampling each with given probability weights."""

    def __init__(self, datasets: list[Dataset], weights: list[float]) -> None:
        assert len(datasets) == len(weights)
        self.datasets = datasets
        total = sum(weights)
        self.weights  = [w / total for w in weights]
        self._len     = sum(len(d) for d in datasets)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ds  = random.choices(self.datasets, weights=self.weights)[0]
        i   = random.randint(0, len(ds) - 1)
        return ds[i]


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    sampler=None,
    num_workers: int = 0,
    **kwargs,
) -> DataLoader:
    """Create a DataLoader with sensible defaults for image deblurring."""
    kw: dict = dict(
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    if num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"]    = 2
    kw.update(kwargs)
    return DataLoader(dataset, **kw)


# ---------------------------------------------------------------------------
# Test / inference dataset
# ---------------------------------------------------------------------------

class Test_Loader(Dataset):
    """Inference-time dataset.  Targets are optional.

    Layout expected::

        data_path/
          input/  *.png
          target/ *.png   ← optional

    Args:
        ddp: If True, exposes a ``get_path`` method (needed by DDP eval scripts).
    """

    def __init__(
        self,
        data_path: str,
        crop_size: int | None = None,
        zero_to_one: bool = False,
        ddp: bool = False,
    ) -> None:
        self.ddp = ddp
        target_dir = os.path.join(data_path, "target")
        self.has_target = os.path.isdir(target_dir)

        self.blur_list  = _glob_images(os.path.join(data_path, "input"))
        self.sharp_list = _glob_images(target_dir) if self.has_target else []

        if self.has_target:
            assert len(self.blur_list) == len(self.sharp_list), (
                "Mismatch between input/ and target/ counts."
            )

        self.transform = _make_transform(crop_size, zero_to_one, augment=False)

    def __len__(self) -> int:
        return len(self.blur_list)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        sample: dict[str, np.ndarray] = {"blur": _read_rgb(self.blur_list[idx])}
        if self.has_target:
            sample["sharp"] = _read_rgb(self.sharp_list[idx])
        result = self.transform(sample)
        result["name"] = os.path.basename(self.blur_list[idx])
        return result

    def get_path(self, idx: int) -> dict[str, str]:
        out = {"blur_path": self.blur_list[idx]}
        if self.has_target:
            out["sharp_path"] = self.sharp_list[idx]
        return out


# Keep old alias for DDP scripts
Test_Loader_DDP = Test_Loader


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_image(path: str, zero_to_one: bool = False) -> torch.Tensor:
    """Load a single image as a (C, H, W) tensor."""
    sample = {"image": _read_rgb(path)}
    sample = Normalize(zero_to_one)(sample)
    sample = ToTensor()(sample)
    return sample["image"]
