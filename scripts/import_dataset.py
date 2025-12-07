#!/usr/bin/env python3
import os
import shutil
import random
from pathlib import Path
import argparse

# ----------------------------------------
# Helpers
# ----------------------------------------

def ensure_dirs(root: Path):
    """Create YOLO directory structure."""
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (root / sub).mkdir(parents=True, exist_ok=True)


def get_next_index(root: Path) -> int:
    """Return next global index by scanning both train/val image folders."""
    imgs = list((root / "images/train").glob("*.jpg")) + \
           list((root / "images/val").glob("*.jpg"))

    if not imgs:
        return 0

    nums = []
    for f in imgs:
        try:
            nums.append(int(f.stem))
        except:
            pass

    return max(nums) + 1 if nums else 0


def write_dataset_yaml(root: Path):
    """Generate dataset.yaml."""
    yaml_path = root / "dataset.yaml"
    yaml_path.write_text(
        "path: data/yolo_pole_dataset\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        "  0: pole\n"
    )
    print(f"[INFO] Wrote dataset.yaml → {yaml_path}")


def clear_yolo_cache(root: Path):
    """Remove YOLO cache files so YOLO rebuilds them."""
    for cache in ["train.cache", "val.cache"]:
        p = root / "labels" / cache
        if p.exists():
            p.unlink()
            print(f"[INFO] Removed cache: {p}")


# ----------------------------------------
# Main import function
# ----------------------------------------

def import_dataset(src_root: Path, dst_root: Path, val_split: float):
    src_imgs = src_root / "images"
    src_labels = src_root / "labels"

    ensure_dirs(dst_root)
    clear_yolo_cache(dst_root)

    # Read + validate image/label pairs
    items = []
    for img in sorted(src_imgs.glob("*.jpg")):
        label = src_labels / f"{img.stem}.txt"
        if not label.exists():
            print(f"[WARNING] Missing label for {img.name}, skipping.")
            continue
        items.append((img, label))

    if not items:
        print("[ERROR] No valid {image,label} pairs found.")
        return

    # Determine splitting
    random.shuffle(items)
    val_count = int(len(items) * val_split)

    val_items = items[:val_count]
    train_items = items[val_count:]

    next_idx = get_next_index(dst_root)
    print(f"[INFO] Starting from index: {next_idx:06d}")
    print(f"[INFO] Adding {len(train_items)} train, {len(val_items)} val samples")

    # Copy function
    def copy_set(pairs, img_dst, lbl_dst, idx_start):
        idx = idx_start
        for img_file, lbl_file in pairs:
            new_name = f"{idx:06d}"
            shutil.copy(img_file, img_dst / f"{new_name}.jpg")
            shutil.copy(lbl_file, lbl_dst / f"{new_name}.txt")
            idx += 1
        return idx

    # Copy train set
    next_idx = copy_set(
        train_items,
        dst_root / "images/train",
        dst_root / "labels/train",
        next_idx
    )

    # Copy val set
    next_idx = copy_set(
        val_items,
        dst_root / "images/val",
        dst_root / "labels/val",
        next_idx
    )

    write_dataset_yaml(dst_root)

    print(f"[DONE] Imported total: {len(items)} samples.")
    print(f"[DONE] Final next index: {next_idx:06d}")


# ----------------------------------------
# Entry point
# ----------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import dataset into YOLO structure.")
    parser.add_argument("src", help="Path to source dataset folder (train_xxxxx)")
    parser.add_argument("--dst", default="data/yolo_pole_dataset",
                        help="Destination YOLO dataset root")
    parser.add_argument("--val", type=float, default=0.20,
                        help="Validation split ratio (default: 0.20)")
    args = parser.parse_args()

    import_dataset(Path(args.src), Path(args.dst), args.val)
