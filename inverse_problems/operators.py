from __future__ import annotations

import torch
import torch.nn.functional as F


class InpaintingOperator:
    def __init__(self, mask: torch.Tensor):
        self.mask = mask

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.mask


class DownsampleOperator:
    def __init__(self, scale: int = 4):
        self.scale = scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return F.interpolate(x, size=(h // self.scale, w // self.scale), mode="bilinear", align_corners=False)


class IdentityOperator:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x
