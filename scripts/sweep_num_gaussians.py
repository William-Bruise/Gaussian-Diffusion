from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
import subprocess
import time
from typing import List


def run_cmd(cmd: List[str]) -> int:
    print("[RUN]", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep Gaussian primitive counts and record reconstruction quality.")
    p.add_argument("--data_root", default="data/ffhq/images")
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--num_gaussians_list", type=int, nargs="+", default=[4096, 8192, 16384])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_val_images", type=int, default=32)
    p.add_argument("--device", default="auto")
    p.add_argument("--encoder_type", choices=["spatial", "global"], default="spatial")
    p.add_argument("--out_csv", default="outputs/sweeps/num_gaussians_sweep.csv")
    args = p.parse_args()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for n in args.num_gaussians_list:
        t0 = time.time()
        run_name = f"N{n}_S{args.image_size}"

        train_cmd = [
            sys.executable,
            "scripts/train_auto_renderer.py",
            "--data_root", args.data_root,
            "--image_size", str(args.image_size),
            "--num_gaussians", str(n),
            "--batch_size", str(args.batch_size),
            "--epochs", str(args.epochs),
            "--run_name", run_name,
            "--encoder_type", args.encoder_type,
        ]
        if args.device != "auto":
            train_cmd += ["--device", args.device]
        rc_train = run_cmd(train_cmd)

        eval_cmd = [
            sys.executable,
            "scripts/eval_reconstruction.py",
            "--data_root", args.data_root,
            "--image_size", str(args.image_size),
            "--num_gaussians", str(n),
            "--encoder_ckpt", f"outputs/checkpoints/{run_name}_encoder.pt",
            "--max_images", str(args.max_val_images),
            "--split", "val",
        ]
        if args.device != "auto":
            eval_cmd += ["--device", args.device]
        rc_eval = run_cmd(eval_cmd)

        # parse latest metrics file emitted by eval script
        metrics_path = Path(f"outputs/eval/{run_name}_recon_metrics.txt")
        l1 = psnr = ssim = float("nan")
        if metrics_path.exists():
            kv = {}
            for line in metrics_path.read_text().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    kv[k.strip()] = float(v.strip())
            l1 = kv.get("l1", float("nan"))
            psnr = kv.get("psnr", float("nan"))
            ssim = kv.get("ssim", float("nan"))

        rows.append({
            "num_gaussians": n,
            "train_rc": rc_train,
            "eval_rc": rc_eval,
            "l1": l1,
            "psnr": psnr,
            "ssim": ssim,
            "elapsed_sec": round(time.time() - t0, 2),
            "run_name": run_name,
            "encoder_type": args.encoder_type,
        })

        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"Sweep done. Results: {out_csv}")


if __name__ == "__main__":
    main()
