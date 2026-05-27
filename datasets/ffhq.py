from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _scan_images(root: Path) -> List[Path]:
    paths = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    return sorted(paths)


@dataclass
class FFHQSplit:
    train: List[Path]
    val: List[Path]


def split_paths(paths: List[Path], val_ratio: float = 0.05) -> FFHQSplit:
    n = len(paths)
    n_val = max(1, int(n * val_ratio)) if n > 1 else 0
    return FFHQSplit(train=paths[:-n_val] if n_val else paths, val=paths[-n_val:] if n_val else [])


class FFHQDataset(Dataset):
    def __init__(self, data_root: str = "data/ffhq/images", split: str = "train", image_size: int = 128, val_ratio: float = 0.05):
        self.root = Path(data_root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"FFHQ directory not found: {self.root}. Please place FFHQ images under {self.root}"
            )
        paths = _scan_images(self.root)
        if len(paths) == 0:
            raise RuntimeError(
                f"No images found in {self.root}. Expected png/jpg/jpeg/webp. Please put FFHQ images there."
            )
        split_paths_obj = split_paths(paths, val_ratio=val_ratio)
        if split == "train":
            self.paths = split_paths_obj.train
        elif split == "val":
            self.paths = split_paths_obj.val
        else:
            raise ValueError("split must be train or val")

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, str(path)
