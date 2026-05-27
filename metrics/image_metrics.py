from __future__ import annotations

import torch


def l1(x, y):
    return (x - y).abs().mean().item()


def psnr(x, y):
    mse = ((x - y) ** 2).mean().clamp_min(1e-8)
    return float(-10.0 * torch.log10(mse))
