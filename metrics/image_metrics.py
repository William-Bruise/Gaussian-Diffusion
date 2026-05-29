from __future__ import annotations

import torch
import torch.nn.functional as F


def l1(x, y):
    return (x - y).abs().mean().item()


def psnr(x, y):
    mse = ((x - y) ** 2).mean().clamp_min(1e-8)
    return float(-10.0 * torch.log10(mse))


def ssim(x, y, c1: float = 0.01**2, c2: float = 0.03**2):
    # simple global SSIM approximation
    mu_x = x.mean(dim=(-1, -2), keepdim=True)
    mu_y = y.mean(dim=(-1, -2), keepdim=True)
    sigma_x = ((x - mu_x) ** 2).mean(dim=(-1, -2), keepdim=True)
    sigma_y = ((y - mu_y) ** 2).mean(dim=(-1, -2), keepdim=True)
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean(dim=(-1, -2), keepdim=True)
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return float((num / den).mean().item())
