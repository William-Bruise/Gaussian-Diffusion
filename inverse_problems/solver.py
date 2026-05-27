from __future__ import annotations

import torch


def solve_inverse_problem(y, operator, renderer, init_z, prior_energy=None, iters=300, lr=1e-2, lambda_prior=1e-2):
    z = init_z.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(iters):
        x = renderer(z, y.shape[-2], y.shape[-1])
        y_hat = operator(x)
        obs = ((y_hat - y) ** 2).mean()
        p = prior_energy(z) if prior_energy is not None else 0.0
        loss = obs + lambda_prior * p
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        x = renderer(z, y.shape[-2], y.shape[-1])
    return z.detach(), x.detach()
