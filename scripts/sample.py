from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import torch

from diffusion.gaussian_ddpm import PrimitiveDenoiser, GaussianDDPM
from models.gaussian_params import GaussianParamConfig, GaussianParametrization
from renderers.gaussian_renderer import GaussianRenderer, visualize_gaussians
from scripts.common import save_tensor_image


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--diffusion_ckpt', required=True)
    p.add_argument('--num_samples', type=int, default=1)
    p.add_argument('--num_gaussians', type=int, default=1024)
    p.add_argument('--resolutions', type=int, nargs='+', default=[128,256,512])
    args = p.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ddpm = GaussianDDPM(PrimitiveDenoiser(raw_dim=8)).to(device)
    ddpm.load_state_dict(torch.load(args.diffusion_ckpt, map_location=device)['ddpm'])
    ddpm.eval()
    renderer = GaussianRenderer().to(device)
    gp = GaussianParametrization(GaussianParamConfig(num_gaussians=args.num_gaussians))

    for i in range(args.num_samples):
        z = ddpm.sample((1, args.num_gaussians, 8), device)
        params = gp.denormalize(z)
        for r in args.resolutions:
            img = renderer(params, r, r)
            save_tensor_image(img.cpu(), f'outputs/samples/sample_{i:03d}_{r}.png')
        visualize_gaussians(params.cpu(), f'outputs/samples/sample_{i:03d}_overlay.png', image_size=max(args.resolutions))


if __name__ == '__main__':
    main()
