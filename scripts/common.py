from __future__ import annotations

from pathlib import Path
import torch
from torchvision.utils import save_image


def denorm_img(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1) * 0.5


def save_tensor_image(x: torch.Tensor, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_image(denorm_img(x), path)
