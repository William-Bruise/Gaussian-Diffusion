from __future__ import annotations

import torch
import torch.nn as nn


class GaussianEncoder(nn.Module):
    def __init__(self, num_gaussians: int = 1024, raw_dim: int = 8, base_ch: int = 64):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.raw_dim = raw_dim
        self.backbone = nn.Sequential(
            nn.Conv2d(3, base_ch, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(base_ch * 4, num_gaussians * raw_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x).flatten(1)
        raw = self.head(h).view(x.shape[0], self.num_gaussians, self.raw_dim)
        return raw
