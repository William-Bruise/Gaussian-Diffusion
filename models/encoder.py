from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _logit(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x) - torch.log1p(-x)


class GaussianEncoder(nn.Module):
    """Global encoder kept for compatibility with old checkpoints.

    This architecture compresses the whole image to one vector before predicting all
    primitives. It is lightweight, but it becomes a bottleneck for large N.
    Prefer SpatialGaussianEncoder for reconstruction sweeps and new training runs.
    """

    def __init__(self, num_gaussians: int = 4096, raw_dim: int = 8, base_ch: int = 64):
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


class SpatialGaussianEncoder(nn.Module):
    """Local, grid-anchored Gaussian encoder for high primitive counts.

    Instead of predicting N independent centers from a single global vector, this
    encoder predicts one primitive per cell on a regular grid. The network only
    predicts a small local center offset plus scale/opacity/color, then converts
    the anchored constrained centers back to raw logits so the existing
    GaussianParametrization.constrain path remains unchanged.
    """

    def __init__(
        self,
        num_gaussians: int = 4096,
        raw_dim: int = 8,
        base_ch: int = 64,
        scale_min: float = 0.01,
        scale_max: float = 0.5,
    ):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.raw_dim = raw_dim
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.grid_size = math.ceil(math.sqrt(num_gaussians))
        self.backbone = nn.Sequential(
            nn.Conv2d(3, base_ch, 3, 1, 1), nn.SiLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, 1, 1), nn.SiLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, 1, 1), nn.SiLU(inplace=True),
        )
        # offset_xy, scale_xy_delta, alpha_raw, rgb_raw = 8 channels
        self.head = nn.Conv2d(base_ch, raw_dim, 1)
        init_scale = min(0.08, max(scale_min * 1.5, 1.5 / self.grid_size))
        scale_unit = (init_scale - scale_min) / (scale_max - scale_min)
        self.register_buffer("scale_bias", torch.tensor(float(math.log(scale_unit / (1.0 - scale_unit)))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        h = self.backbone(x)
        h = F.interpolate(h, size=(self.grid_size, self.grid_size), mode="bilinear", align_corners=False)
        raw_map = self.head(h).permute(0, 2, 3, 1).reshape(b, self.grid_size * self.grid_size, self.raw_dim)
        raw_map = raw_map[:, : self.num_gaussians]

        ys = torch.linspace(0.5 / self.grid_size, 1.0 - 0.5 / self.grid_size, self.grid_size, device=x.device)
        xs = torch.linspace(0.5 / self.grid_size, 1.0 - 0.5 / self.grid_size, self.grid_size, device=x.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        base = torch.stack([xx, yy], dim=-1).reshape(1, self.grid_size * self.grid_size, 2)[:, : self.num_gaussians]
        offset = torch.tanh(raw_map[..., 0:2]) * (0.45 / self.grid_size)
        center = (base + offset).clamp(1e-4, 1.0 - 1e-4)

        raw = raw_map.clone()
        raw[..., 0:2] = _logit(center)
        raw[..., 2:4] = raw[..., 2:4] + self.scale_bias
        return raw


def build_gaussian_encoder(
    encoder_type: str = "spatial",
    num_gaussians: int = 4096,
    raw_dim: int = 8,
    base_ch: int = 64,
    scale_min: float = 0.01,
    scale_max: float = 0.5,
) -> nn.Module:
    if encoder_type == "global":
        return GaussianEncoder(num_gaussians=num_gaussians, raw_dim=raw_dim, base_ch=base_ch)
    if encoder_type == "spatial":
        return SpatialGaussianEncoder(
            num_gaussians=num_gaussians,
            raw_dim=raw_dim,
            base_ch=base_ch,
            scale_min=scale_min,
            scale_max=scale_max,
        )
    raise ValueError(f"Unknown encoder_type: {encoder_type}")
