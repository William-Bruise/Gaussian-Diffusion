from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class PrimitiveDenoiser(nn.Module):
    def __init__(self, d_model: int = 128, raw_dim: int = 8, depth: int = 4, nhead: int = 8):
        super().__init__()
        self.in_proj = nn.Linear(raw_dim, d_model)
        self.t_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.net = nn.TransformerEncoder(enc, num_layers=depth)
        self.out = nn.Linear(d_model, raw_dim)
        self.d_model = d_model

    def forward(self, zt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(zt)
        te = self.t_proj(timestep_embedding(t, self.d_model)).unsqueeze(1)
        h = self.net(h + te)
        return self.out(h)


class GaussianDDPM(nn.Module):
    def __init__(self, denoiser: PrimitiveDenoiser, timesteps: int = 1000, beta1: float = 1e-4, beta2: float = 0.02):
        super().__init__()
        self.denoiser = denoiser
        self.timesteps = timesteps
        betas = torch.linspace(beta1, beta2, timesteps)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("abar", abar)

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor | None = None):
        if eps is None:
            eps = torch.randn_like(z0)
        a = self.abar[t].view(-1, 1, 1)
        zt = torch.sqrt(a) * z0 + torch.sqrt(1 - a) * eps
        return zt, eps

    def loss(self, z0: torch.Tensor):
        b = z0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=z0.device)
        zt, eps = self.q_sample(z0, t)
        pred = self.denoiser(zt, t)
        return F.mse_loss(pred, eps)

    @torch.no_grad()
    def sample(self, shape, device):
        z = torch.randn(shape, device=device)
        for ti in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), ti, device=device, dtype=torch.long)
            eps = self.denoiser(z, t)
            a = self.alphas[ti]
            abar = self.abar[ti]
            beta = self.betas[ti]
            mean = (z - (beta / torch.sqrt(1 - abar)) * eps) / torch.sqrt(a)
            if ti > 0:
                z = mean + torch.sqrt(beta) * torch.randn_like(z)
            else:
                z = mean
        return z
