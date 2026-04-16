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
"""

from __future__ import annotations

import glob
import os
import random
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
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


def _make_transform(crop_size: int | None, zero_to_one: bool, augment: bool) -> Callable:
    steps: list[Callable] = []
    if crop_size:
        steps.append(RandomCrop(crop_size, crop_size))
    if augment:
        steps.append(RandomFlip())
        steps.append(RandomRotate())
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
        augment: bool = True,
    ) -> None:
        self.blur_list: list[str] = []
        self.sharp_list: list[str] = []
        self._collect_lists(data_path, mode)
        assert len(self.blur_list) == len(self.sharp_list), (
            f"Length mismatch: {len(self.blur_list)} blur vs {len(self.sharp_list)} sharp"
        )
        self.transform = _make_transform(crop_size, zero_to_one, augment and mode == "train")

    def _collect_lists(self, data_path: str, mode: str) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.blur_list)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
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
