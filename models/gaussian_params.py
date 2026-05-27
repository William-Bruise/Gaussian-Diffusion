from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GaussianParamConfig:
    num_gaussians: int = 256
    raw_dim: int = 8
    scale_min: float = 0.01
    scale_max: float = 0.5


class GaussianParametrization:
    def __init__(self, cfg: GaussianParamConfig):
        self.cfg = cfg

    def constrain(self, raw: torch.Tensor) -> torch.Tensor:
        # raw: B x N x 8
        center = torch.sigmoid(raw[..., 0:2])
        scale = self.cfg.scale_min + torch.sigmoid(raw[..., 2:4]) * (self.cfg.scale_max - self.cfg.scale_min)
        alpha = torch.sigmoid(raw[..., 4:5])
        color = torch.tanh(raw[..., 5:8])
        return torch.cat([center, scale, alpha, color], dim=-1)

    def normalize(self, params: torch.Tensor) -> torch.Tensor:
        # params already constrained. map to roughly [-1,1] for DDPM
        out = params.clone()
        out[..., 0:2] = params[..., 0:2] * 2 - 1
        out[..., 2:4] = ((params[..., 2:4] - self.cfg.scale_min) / (self.cfg.scale_max - self.cfg.scale_min)) * 2 - 1
        out[..., 4:5] = params[..., 4:5] * 2 - 1
        # color already [-1,1]
        return out

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        out = z.clone()
        out[..., 0:2] = (z[..., 0:2] + 1) * 0.5
        out[..., 2:4] = (z[..., 2:4] + 1) * 0.5 * (self.cfg.scale_max - self.cfg.scale_min) + self.cfg.scale_min
        out[..., 4:5] = (z[..., 4:5] + 1) * 0.5
        out[..., 5:8] = torch.clamp(z[..., 5:8], -1, 1)
        return out
