from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class GaussianRenderer(nn.Module):
    def __init__(self, eps: float = 1e-6, chunk_size: int = 64):
        super().__init__()
        self.eps = eps
        self.chunk_size = chunk_size

    def forward(self, params: torch.Tensor, height: int, width: int, normalize: bool = True) -> torch.Tensor:
        # params: B,N,8 constrained
        b, n, _ = params.shape
        device = params.device
        ys = torch.linspace(0, 1, height, device=device)
        xs = torch.linspace(0, 1, width, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([xx, yy], dim=-1).view(1, 1, height, width, 2)

        center = params[..., 0:2].view(b, n, 1, 1, 2)
        scale = params[..., 2:4].view(b, n, 1, 1, 2).clamp_min(1e-4)
        alpha = params[..., 4:5].view(b, n, 1, 1, 1)
        color = params[..., 5:8].view(b, n, 1, 1, 3)

        numerator = torch.zeros(b, height, width, 3, device=device)
        denom = torch.zeros(b, height, width, 1, device=device)
        for i in range(0, n, self.chunk_size):
            j = min(i + self.chunk_size, n)
            d = (coords - center[:, i:j]) / scale[:, i:j]
            g = torch.exp(-0.5 * (d[..., 0] ** 2 + d[..., 1] ** 2)).unsqueeze(-1)
            w = alpha[:, i:j] * g
            numerator = numerator + (w * color[:, i:j]).sum(dim=1)
            denom = denom + w.sum(dim=1)

        img = numerator / (denom + self.eps) if normalize else numerator
        img = img.permute(0, 3, 1, 2).clamp(-1, 1)
        return img


def visualize_gaussians(params: torch.Tensor, save_path: str, image_size: int = 256, max_draw: int = 128):
    p = params[0].detach().cpu()
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xlim([0, image_size])
    ax.set_ylim([image_size, 0])
    ax.set_facecolor("black")
    for g in p[:max_draw]:
        cx, cy, sx, sy, a = g[:5].tolist()
        ax.add_patch(plt.Circle((cx * image_size, cy * image_size), radius=max(1, sx * image_size), color="lime", alpha=float(a)*0.2))
    ax.set_title("Gaussian centers/scales overlay")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
