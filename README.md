# Continuous-Resolution Gaussian Parameter Diffusion (Prototype)

Minimal runnable research prototype for FFHQ priors with diffusion in Gaussian primitive parameter space.

## Project structure
`configs/ datasets/ models/ renderers/ diffusion/ trainers/ samplers/ inverse_problems/ metrics/ scripts/ docs/`

## FFHQ check
```bash
python scripts/check_ffhq.py --data_root /home/wuweihao/Datasets/FFHQ
```
If directory does not exist or has no images, script raises clear errors and does not download anything.

## Stage 1: train auto-renderer
```bash
python scripts/train_auto_renderer.py \
  --data_root /home/wuweihao/Datasets/FFHQ \
  --image_size 128 \
  --num_gaussians 256 \
  --batch_size 8
```

## Stage 2: train Gaussian parameter diffusion
```bash
python scripts/train_gaussian_diffusion.py \
  --data_root /home/wuweihao/Datasets/FFHQ \
  --encoder_ckpt outputs/checkpoints/encoder.pt \
  --num_gaussians 256
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
