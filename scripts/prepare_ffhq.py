from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import shutil

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def scan_images(root: Path):
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS])


def main():
    p = argparse.ArgumentParser(description="Move/copy FFHQ images into this project under data/ffhq/images")
    p.add_argument("--src_root", default="/home/wuweihao/ArtDiffusion/data/ffhq/images")
    p.add_argument("--dst_root", default=str(PROJECT_ROOT / "data" / "ffhq" / "images"))
    p.add_argument("--mode", choices=["copy", "move"], default="copy")
    args = p.parse_args()

    src = Path(args.src_root)
    dst = Path(args.dst_root)
    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {src}")

    images = scan_images(src)
    if len(images) == 0:
        raise RuntimeError(f"No images found in source: {src}")

    dst.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(images, 1):
        target = dst / s.name
        if target.exists():
            stem, suffix = target.stem, target.suffix
            target = dst / f"{stem}_{i:06d}{suffix}"
        if args.mode == "copy":
            shutil.copy2(s, target)
        else:
            shutil.move(str(s), str(target))

    print(f"Prepared FFHQ images: {len(images)}")
    print(f"Destination: {dst}")


if __name__ == "__main__":
    main()
