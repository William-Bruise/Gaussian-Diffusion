from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from datasets.ffhq import FFHQDataset
from models.encoder import GaussianEncoder
from models.gaussian_params import GaussianParamConfig, GaussianParametrization
from diffusion.gaussian_ddpm import PrimitiveDenoiser, GaussianDDPM
from renderers.gaussian_renderer import GaussianRenderer
from scripts.common import save_tensor_image


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='data/ffhq/images')
    p.add_argument('--encoder_ckpt', required=True)
    p.add_argument('--image_size', type=int, default=128)
    p.add_argument('--num_gaussians', type=int, default=1024)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=1)
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = FFHQDataset(args.data_root, 'train', args.image_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    enc = GaussianEncoder(args.num_gaussians).to(device)
    enc.load_state_dict(torch.load(args.encoder_ckpt, map_location=device)['encoder'])
    enc.eval()
    for p_ in enc.parameters(): p_.requires_grad = False

    gparam = GaussianParametrization(GaussianParamConfig(num_gaussians=args.num_gaussians))
    denoiser = PrimitiveDenoiser(raw_dim=8)
    ddpm = GaussianDDPM(denoiser).to(device)
    opt = torch.optim.Adam(ddpm.parameters(), lr=1e-4)
    renderer = GaussianRenderer().to(device)

    for ep in range(args.epochs):
        for it, (x, _) in enumerate(dl):
            x = x.to(device)
            with torch.no_grad():
                params = gparam.constrain(enc(x))
                z0 = gparam.normalize(params)
            loss = ddpm.loss(z0)
            opt.zero_grad(); loss.backward(); opt.step()
            if it % 50 == 0:
                print(f'ep {ep} it {it} loss {loss.item():.4f}')
                with torch.no_grad():
                    zs = ddpm.sample((1, args.num_gaussians, 8), device)
                    ps = gparam.denormalize(zs)
                    img = renderer(ps, args.image_size, args.image_size)
                    save_tensor_image(img.cpu(), f'outputs/samples/train_ep{ep}_it{it}.png')

    Path('outputs/checkpoints').mkdir(parents=True, exist_ok=True)
    torch.save({'ddpm': ddpm.state_dict(), 'num_gaussians': args.num_gaussians}, 'outputs/checkpoints/diffusion.pt')


if __name__ == '__main__':
    main()
