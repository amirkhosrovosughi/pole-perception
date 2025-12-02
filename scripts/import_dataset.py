#!/usr/bin/env python3
import os
import shutil
from pathlib import Path
import argparse

def ensure_dirs(root):
    (root / "images/train").mkdir(parents=True, exist_ok=True)
    (root / "labels/train").mkdir(parents=True, exist_ok=True)

def get_next_index(folder):
    """Return the next file index based on existing files."""
    files = list(folder.glob("*.jpg"))
    if not files:
        return 0
    nums = []
    for f in files:
        try:
            nums.append(int(f.stem))
        except:
            pass
    return max(nums) + 1 if nums else 0

def import_dataset(src_root, dst_root):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    ensure_dirs(dst_root)

    src_imgs = src_root / "images"
    src_labels = src_root / "labels"

    dst_imgs = dst_root / "images/train"
    dst_labels = dst_root / "labels/train"

    next_idx = get_next_index(dst_imgs)
    print(f"Starting from index: {next_idx}")

    added = 0

    for img_file in sorted(src_imgs.glob("*.jpg")):
        base = img_file.stem
        label_file = src_labels / f"{base}.txt"

        if not label_file.exists():
            print(f"WARNING: Missing label for {img_file.name}, skipping.")
            continue

        new_name = f"{next_idx:06d}"
        shutil.copy(img_file, dst_imgs / f"{new_name}.jpg")
        shutil.copy(label_file, dst_labels / f"{new_name}.txt")

        next_idx += 1
        added += 1

    print(f"Imported {added} samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="Path to input dataset folder (train_xxxxxx)")
    parser.add_argument("--dst", default="data/yolo_pole_dataset",
                        help="Destination YOLO dataset root")
    args = parser.parse_args()

    import_dataset(args.src, Path(args.dst))
