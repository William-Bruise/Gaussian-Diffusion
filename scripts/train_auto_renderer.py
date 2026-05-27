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
from renderers.gaussian_renderer import GaussianRenderer, visualize_gaussians
from scripts.common import save_tensor_image


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='data/ffhq/images')
    p.add_argument('--image_size', type=int, default=128)
    p.add_argument('--num_gaussians', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4)
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = FFHQDataset(args.data_root, 'train', args.image_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    enc = GaussianEncoder(args.num_gaussians).to(device)
    renderer = GaussianRenderer().to(device)
    gparam = GaussianParametrization(GaussianParamConfig(num_gaussians=args.num_gaussians))
    opt = torch.optim.Adam(enc.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        for it, (x, _) in enumerate(dl):
            x = x.to(device)
            raw = enc(x)
            params = gparam.constrain(raw)
            xhat = renderer(params, args.image_size, args.image_size)
            l_rec = (xhat - x).abs().mean()
            l_scale = params[..., 2:4].mean()
            l_alpha = params[..., 4:5].mean()
            loss = l_rec + 1e-3 * l_scale + 1e-3 * l_alpha
            opt.zero_grad(); loss.backward(); opt.step()
            if it % 50 == 0:
                print(f'ep {ep} it {it} loss {loss.item():.4f}')
                save_tensor_image(xhat[:1].cpu(), f'outputs/recon/ep{ep}_it{it}.png')
                visualize_gaussians(params[:1], f'outputs/recon/ep{ep}_it{it}_overlay.png', args.image_size)

    Path('outputs/checkpoints').mkdir(parents=True, exist_ok=True)
    torch.save({'encoder': enc.state_dict(), 'num_gaussians': args.num_gaussians}, 'outputs/checkpoints/encoder.pt')


if __name__ == '__main__':
    main()
