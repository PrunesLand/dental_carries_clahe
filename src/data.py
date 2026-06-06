"""
Dataset utilities: file listing, RAM cache (raw + CLAHE), and PyTorch Dataset.
"""

import os
import random

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

import config


def list_files() -> np.ndarray:
    """Return a sorted array of image filenames found in IMAGE_PATH."""
    return np.array(sorted(
        f for f in os.listdir(config.IMAGE_PATH)
        if f.endswith((".png", ".jpg"))
    ))


class ImageCache:
    """
    Loads every image/label once, resizes to IMAGE_SIZE, and stores both
    raw and CLAHE versions in RAM.

    Eliminates per-epoch disk I/O and repeated deterministic CLAHE
    computation — a significant speedup for small datasets.
    """

    def __init__(self, file_names: np.ndarray):
        H, W = config.IMAGE_SIZE
        clahe_op = cv2.createCLAHE(
            clipLimit=config.CLIP_LIMIT,
            tileGridSize=config.GRID_SIZE,
        )
        self.raw: dict[str, np.ndarray] = {}
        self.clahe: dict[str, np.ndarray] = {}
        self.label: dict[str, np.ndarray] = {}

        for f in file_names:
            img = np.array(Image.open(config.IMAGE_PATH + f).convert("L"))
            lbl = np.array(Image.open(config.LABEL_PATH + f).convert("L"))
            img = cv2.resize(img, (W, H))
            lbl = cv2.resize(lbl, (W, H), interpolation=cv2.INTER_NEAREST)
            self.raw[f] = img
            self.clahe[f] = clahe_op.apply(img)
            self.label[f] = lbl

        print(f"Cached {len(file_names)} images in RAM (raw + CLAHE).")


class CariesDataset(Dataset):
    """
    Indexes into an ImageCache.  CLAHE is just a dict lookup; only the
    random geometric augmentation runs per __getitem__.

    Tensors stay on CPU so num_workers > 0 is safe.
    """

    def __init__(
        self,
        cache: ImageCache,
        file_names: np.ndarray,
        augmentation: bool = False,
        use_clahe: bool = False,
    ):
        self.cache = cache
        self.files = list(file_names)
        self.augmentation = augmentation
        self.use_clahe = use_clahe

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        f = self.files[idx]
        img_np = self.cache.clahe[f] if self.use_clahe else self.cache.raw[f]
        img = Image.fromarray(img_np)
        lbl = Image.fromarray(self.cache.label[f])

        if self.augmentation:
            if random.random() < 0.5:
                img = TF.hflip(img)
                lbl = TF.hflip(lbl)
            angle = random.uniform(-5, 5)
            img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)
            lbl = TF.rotate(lbl, angle, interpolation=InterpolationMode.NEAREST)

        img = TF.to_tensor(img)                    # [1, H, W] in [0, 1]
        lbl = (TF.to_tensor(lbl) > 0.5).float()
        return img, lbl
