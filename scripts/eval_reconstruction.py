import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse, torch
from metrics.image_metrics import l1, psnr

p=argparse.ArgumentParser();p.add_argument('--pred',required=True);p.add_argument('--gt',required=True);args=p.parse_args()
pred=torch.load(args.pred);gt=torch.load(args.gt)
print({'l1':l1(pred,gt),'psnr':psnr(pred,gt)})
