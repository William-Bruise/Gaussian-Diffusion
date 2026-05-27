import argparse
from datasets.ffhq import FFHQDataset

p = argparse.ArgumentParser()
p.add_argument('--data_root', default='/home/wuweihao/Datasets/FFHQ')
p.add_argument('--image_size', type=int, default=128)
args = p.parse_args()

ds = FFHQDataset(args.data_root, 'train', args.image_size)
print(f'train images: {len(ds)}')
print('ok')
