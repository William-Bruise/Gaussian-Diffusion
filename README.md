# Continuous-Resolution Gaussian Parameter Diffusion (Prototype)

Minimal runnable research prototype for FFHQ priors with diffusion in Gaussian primitive parameter space.

## Project structure
`configs/ datasets/ models/ renderers/ diffusion/ trainers/ samplers/ inverse_problems/ metrics/ scripts/ docs/`

## Prepare local FFHQ into this project
```bash
python scripts/prepare_ffhq.py \
  --src_root /home/wuweihao/ArtDiffusion/data/ffhq/images \
  --dst_root data/ffhq/images \
  --mode copy
```

## FFHQ check
```bash
python scripts/check_ffhq.py --data_root data/ffhq/images
```
If directory does not exist or has no images, script raises clear errors and does not download anything.

## Stage 1: train auto-renderer
```bash
python scripts/train_auto_renderer.py \
  --data_root data/ffhq/images \
  --image_size 128 \
  --num_gaussians 4096 \
  --batch_size 4 \
  --encoder_type spatial
```

## Stage 2: train Gaussian parameter diffusion
```bash
python scripts/train_gaussian_diffusion.py \
  --data_root data/ffhq/images \
  --encoder_ckpt outputs/checkpoints/encoder.pt \
  --num_gaussians 4096
```

## Sampling (multi-resolution from same Gaussian params)
```bash
python scripts/sample.py \
  --diffusion_ckpt outputs/checkpoints/diffusion.pt \
  --resolutions 128 256 512
```

## Inverse problems
```bash
python scripts/run_inverse.py \
  --task inpainting \
  --image_path path/to/test.png \
  --diffusion_ckpt outputs/checkpoints/diffusion.pt \
  --output_resolution 256
```

Supports `inpainting`, `super-resolution`, `denoising` with:
- renderer-only baseline
- renderer + diffusion-prior-inspired baseline


## Recommended primitive count
For 128x128 FFHQ, `4096` can still underfit if the encoder is weak or under-trained. New runs use `--encoder_type spatial`, which anchors primitives on a local grid; sweep `4096/8192/16384` and choose by validation PSNR/SSIM rather than guessing.

## Sweep num_gaussians by experiment
```bash
python scripts/sweep_num_gaussians.py \
  --data_root data/ffhq/images \
  --image_size 128 \
  --num_gaussians_list 4096 8192 16384 \
  --epochs 5 \
  --max_val_images 32 \
  --encoder_type spatial
```
Outputs CSV: `outputs/sweeps/num_gaussians_sweep.csv`
