from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import torch
from torch.utils.data import DataLoader

from datasets.ffhq import FFHQDataset
from metrics.image_metrics import l1 as l1_metric, psnr as psnr_metric, ssim as ssim_metric
from models.encoder import GaussianEncoder
from models.gaussian_params import GaussianParamConfig, GaussianParametrization
from renderers.gaussian_renderer import GaussianRenderer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='data/ffhq/images')
    p.add_argument('--split', default='val', choices=['train', 'val'])
    p.add_argument('--image_size', type=int, default=128)
    p.add_argument('--num_gaussians', type=int, default=1024)
    p.add_argument('--encoder_ckpt', required=True)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--max_images', type=int, default=32)
    p.add_argument('--device', default='auto')
    args = p.parse_args()

    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    ds = FFHQDataset(args.data_root, args.split, args.image_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    enc = GaussianEncoder(args.num_gaussians).to(device)
    ckpt = torch.load(args.encoder_ckpt, map_location=device)
    enc.load_state_dict(ckpt['encoder'])
    enc.eval()

    gp = GaussianParametrization(GaussianParamConfig(num_gaussians=args.num_gaussians))
    renderer = GaussianRenderer().to(device)

    acc_l1 = acc_psnr = acc_ssim = 0.0
    seen = 0
    with torch.no_grad():
        for x, _ in dl:
            x = x.to(device)
            params = gp.constrain(enc(x))
            x_hat = renderer(params, args.image_size, args.image_size)
            b = x.shape[0]
            for i in range(b):
                acc_l1 += l1_metric(x_hat[i:i+1], x[i:i+1])
                acc_psnr += psnr_metric(x_hat[i:i+1], x[i:i+1])
                acc_ssim += ssim_metric(x_hat[i:i+1], x[i:i+1])
            seen += b
            if seen >= args.max_images:
                break

    denom = max(1, min(seen, args.max_images))
    out = {
        'l1': acc_l1 / denom,
        'psnr': acc_psnr / denom,
        'ssim': acc_ssim / denom,
    }
    run_name = Path(args.encoder_ckpt).stem.replace('_encoder', '')
    out_dir = Path('outputs/eval')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{run_name}_recon_metrics.txt'
    out_path.write_text('\n'.join([f'{k}: {v:.6f}' for k, v in out.items()]) + '\n')
    print(out)
    print(f'saved: {out_path}')


if __name__ == '__main__':
    main()
