from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from PIL import Image
import torch
from torchvision import transforms

from diffusion.gaussian_ddpm import PrimitiveDenoiser, GaussianDDPM
from models.gaussian_params import GaussianParamConfig, GaussianParametrization
from renderers.gaussian_renderer import GaussianRenderer
from inverse_problems.operators import InpaintingOperator, DownsampleOperator, IdentityOperator
from inverse_problems.solver import solve_inverse_problem
from scripts.common import save_tensor_image


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=['inpainting','super-resolution','denoising'], required=True)
    p.add_argument('--image_path', required=True)
    p.add_argument('--diffusion_ckpt', required=True)
    p.add_argument('--output_resolution', type=int, default=256)
    p.add_argument('--num_gaussians', type=int, default=1024)
    args = p.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x = transforms.Compose([transforms.Resize(args.output_resolution), transforms.CenterCrop(args.output_resolution), transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)])(Image.open(args.image_path).convert('RGB')).unsqueeze(0).to(device)
    renderer = GaussianRenderer().to(device)

    if args.task == 'inpainting':
        mask = torch.ones_like(x); mask[..., args.output_resolution//4:args.output_resolution//2, args.output_resolution//4:args.output_resolution//2] = 0
        op = InpaintingOperator(mask)
    elif args.task == 'super-resolution':
        op = DownsampleOperator(scale=4)
    else:
        op = IdentityOperator(); x = x + 0.1*torch.randn_like(x)
    y = op(x)

    ddpm = GaussianDDPM(PrimitiveDenoiser(raw_dim=8)).to(device)
    ddpm.load_state_dict(torch.load(args.diffusion_ckpt, map_location=device)['ddpm']); ddpm.eval()
    gp = GaussianParametrization(GaussianParamConfig(num_gaussians=args.num_gaussians))

    z_init = gp.denormalize(ddpm.sample((1,args.num_gaussians,8), device)).detach()
    prior_energy = lambda z: (z ** 2).mean()
    z_prior, x_prior = solve_inverse_problem(y, op, renderer, z_init, prior_energy=prior_energy, iters=200)
    z_base, x_base = solve_inverse_problem(y, op, renderer, z_init, prior_energy=None, iters=200)

    save_tensor_image(x.cpu(), 'outputs/inverse/target.png')
    save_tensor_image(x_base.cpu(), 'outputs/inverse/recon_renderer_only.png')
    save_tensor_image(x_prior.cpu(), 'outputs/inverse/recon_renderer_plus_prior.png')

if __name__ == '__main__':
    main()
