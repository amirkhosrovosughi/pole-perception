#!/usr/bin/env python3
"""
parse_bag_to_yolo.py

Reads a ros2 bag (using rosbags), extracts images and generates YOLO-format
labels using one of two methods:

1) Preferred (if available):
   - Uses bounding boxes from `/featureDetection/bbox`
   - Message type: vision_msgs::msg::Detection3DArray
   - Uses 2D bounding box info only (x, y, size.x, size.y)
   - Values are already normalized in [0, 1]
   - No metadata or odometry required

2) Fallback (legacy behavior):
   - Uses PX4 vehicle odometry
   - Uses camera + pole metadata
   - Projects 3D pole geometry into the image to compute bounding boxes

Outputs:
- images/
- labels/          (YOLO txt)
- annotation/      (CVAT-compatible structure)
- meta3d/          (optional JSON metadata, odom mode only)

Usage:
  python scripts/parse_bag_to_yolo.py \
    --bag bags/2025_11_06-07_23_41/rosbag2_2025_11_06-07_23_41 \
    --metadata bags/2025_11_06-07_23_41/metadata.xml \
    --output data/train \
    --topic-image /camera \
    --topic-odom /fmu/out/vehicle_odometry
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

# Local module:
# - loads camera / pole metadata
# - computes bounding boxes from odometry
from label_from_odometry import (
    load_metadata,
    compute_bbox_from_odom,
    is_metadata_loaded,
)

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

# --------------------------
# Bag reader availability
# --------------------------
_HAS_ROSBAGS = False

try:
    # Modern rosbags API
    from rosbags.highlevel import AnyReader
    _HAS_ROSBAGS = True
except ImportError:
    try:
        # Legacy rosbags API
        from rosbags.rosbag2 import Reader
        from rosbags.typesys import deserialize_cdr
        _HAS_ROSBAGS = True
    except ImportError:
        pass

if not _HAS_ROSBAGS:
    print("❌ 'rosbags' not available. Install with: pip install rosbags")
    sys.exit(1)

# --------------------------
# Configurable defaults
# --------------------------
DEFAULT_LINEAR_VEL_THRESH = 2.0     # m/s
DEFAULT_ANGULAR_VEL_THRESH = 1.0    # rad/s
DEFAULT_SAMPLE_STRIDE = 5           # Keep every Nth frame
CLASS_ID = 0                        # Single-class (pole)

# Preferred bounding box topic
BBOX_TOPIC = "/featureDetection/bbox"


def create_cvat_annotation_zip(cvat_dir, out_dir) -> Path:
    """
    Creates a CVAT-compatible annotation.zip file from the generated dataset.
    """
    from pathlib import Path
    import zipfile

    # Ensure Path types
    cvat_dir = Path(cvat_dir)
    out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = out_dir / "annotation.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in cvat_dir.rglob("*"):
            zf.write(file, file.relative_to(cvat_dir.parent))

    print(f"CVAT annotation zip created: {zip_path}")
    return zip_path


def process_with_rosbags(
    bag_dir,
    metadata_path,
    out_dir,
    topic_image,
    topic_odom,
    lin_thresh,
    ang_thresh,
    frame_stride,
    save_json=True,
):
    """
    Main processing routine.

    - Iterates through rosbag messages
    - Detects whether /featureDetection/bbox exists
    - Selects labeling strategy automatically
    - Writes YOLO labels, images, and optional metadata
    """
    from rosbags.rosbag2 import Reader
    from rosbags.serde import deserialize_cdr
    from rosbags.typesys import get_types_from_msg
    from rosbags.typesys import Stores, get_typestore
    from pathlib import Path

    # Create explicit TypeStore
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    # --------------------------------------------------
    # Register message definitions (local + system)
    # --------------------------------------------------
    all_types = {}

    # Local msg definitions (if any)
    msg_dir = Path("msg")
    if msg_dir.exists():
        for msg_file in msg_dir.glob("*.msg"):
            pkg = "px4_msgs"
            types = get_types_from_msg(
                msg_file.read_text(),
                f"{pkg}/msg/{msg_file.stem}",
            )
            all_types.update(types)

    # System-installed vision_msgs (REQUIRED)
    vision_msgs_path = Path("/opt/ros/humble/share/vision_msgs/msg")
    if vision_msgs_path.exists():
        for msg_file in vision_msgs_path.glob("*.msg"):
            types = get_types_from_msg(
                msg_file.read_text(),
                f"vision_msgs/msg/{msg_file.stem}",
            )
            all_types.update(types)

    # Register all collected types
    if all_types:
        typestore.register(all_types)

    # --------------------------------------------------
    # Output directory structure
    # --------------------------------------------------
    images_dir = Path(out_dir) / "images"
    labels_dir = Path(out_dir) / "labels"
    json_dir = Path(out_dir) / "meta3d"

    cvat_dir = Path(out_dir) / "annotation"
    images_cvat_dir = cvat_dir / "obj_train_data" / "images"

    for d in [images_dir, labels_dir, images_cvat_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if save_json:
        json_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # CVAT boilerplate files
    # --------------------------------------------------
    (cvat_dir / "obj.names").write_text("pole\n")
    (cvat_dir / "obj.data").write_text(
        "classes = 1\n"
        "train = train.txt\n"
        "names = obj.names\n"
        "backup = backup/\n"
    )

    train_file = cvat_dir / "train.txt"
    train_lines = train_file.read_text().splitlines() if train_file.exists() else []

    # --------------------------------------------------
    # Open bag and inspect available topics
    # --------------------------------------------------
    reader = Reader(bag_dir)
    reader.open()

    topics = {c.topic for c in reader.connections}

    # Decide labeling mode
    use_bbox_topic = BBOX_TOPIC in topics

    if use_bbox_topic:
        print(f"✅ Using bounding boxes from topic: {BBOX_TOPIC}")
    else:
        print("⚠️  BBox topic not found. Falling back to odometry-based labeling.")

        if metadata_path is None:
            raise RuntimeError(
                "Metadata file is required when /featureDetection/bbox is not available."
            )

        load_metadata(metadata_path)
        if not is_metadata_loaded():
            raise RuntimeError("Failed to load metadata")

        if topic_odom not in topics:
            raise RuntimeError("Odometry topic not found in bag")

    # --------------------------------------------------
    # Runtime state
    # --------------------------------------------------
    latest_odom = None          # Latest PX4 odometry
    latest_bboxes = None        # Latest Detection3DArray

    frame_idx = 0
    image_counter = 0

    # --------------------------------------------------
    # Main loop
    # --------------------------------------------------
    for conn, timestamp, rawdata in tqdm(reader.messages(), desc="Reading bag"):
        msg = typestore.deserialize_cdr(rawdata, conn.msgtype)

        # --------------------------
        # Odometry update (fallback)
        # --------------------------
        if conn.topic == topic_odom and not use_bbox_topic:
            try:
                latest_odom = {
                    "pos": np.array(msg.position, dtype=float),
                    "q": np.array(msg.q, dtype=float),
                    "v_lin": float(np.linalg.norm(msg.velocity)),
                    "v_ang": float(np.linalg.norm(msg.angular_velocity)),
                }
            except Exception:
                continue

        # --------------------------
        # Bounding box update
        # --------------------------
        elif conn.topic == BBOX_TOPIC and use_bbox_topic:
            latest_bboxes = msg.detections

        # --------------------------
        # Image handling
        # --------------------------
        elif conn.topic == topic_image:
            image_counter += 1
            if image_counter % frame_stride != 0:
                continue

            if use_bbox_topic and latest_bboxes is None:
                continue

            if not use_bbox_topic and latest_odom is None:
                continue

            # Deserialize image
            try:
                h, w = msg.height, msg.width
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, -1)
            except Exception:
                continue

            label_path = labels_dir / f"{frame_idx:06d}.txt"
            cvat_label_path = images_cvat_dir / f"{frame_idx:06d}.txt"

            with open(label_path, "w") as label_f, open(cvat_label_path, "w") as cvat_label_f:

                # --------------------------
                # Mode 1: BBox topic
                # --------------------------
                if use_bbox_topic:
                    for det in latest_bboxes:
                        bb = det.bbox
                        x_c = bb.center.position.x
                        y_c = bb.center.position.y
                        bbox_w = bb.size.x
                        bbox_h = bb.size.y

                        # Defensive check: normalized values
                        if not (0.0 <= x_c <= 1.0 and 0.0 <= y_c <= 1.0):
                            continue

                        label_f.write(f"{CLASS_ID} {x_c:.6f} {y_c:.6f} {bbox_w:.6f} {bbox_h:.6f}\n")
                        cvat_label_f.write(f"{CLASS_ID} {x_c:.6f} {y_c:.6f} {bbox_w:.6f} {bbox_h:.6f}\n")

                # --------------------------
                # Mode 2: Odometry fallback
                # --------------------------
                else:
                    if (
                        latest_odom["v_lin"] > lin_thresh
                        or latest_odom["v_ang"] > ang_thresh
                    ):
                        continue

                    results = compute_bbox_from_odom(
                        latest_odom["pos"],
                        latest_odom["q"],
                        img.shape,
                        debug=False,
                    )

                    for res in results:
                        if not res.get("valid", False):
                            continue

                        x_c, y_c, bbox_w, bbox_h = res["bbox_norm"]
                        label_f.write(f"{CLASS_ID} {x_c:.6f} {y_c:.6f} {bbox_w:.6f} {bbox_h:.6f}\n")
                        cvat_label_f.write(f"{CLASS_ID} {x_c:.6f} {y_c:.6f} {bbox_w:.6f} {bbox_h:.6f}\n")

            # Save image
            img_name = f"{frame_idx:06d}.jpg"
            cv2.imwrite(str(images_dir / img_name), img)
            cv2.imwrite(str(images_cvat_dir / img_name), img)

            # Update train.txt
            train_lines.append(str((images_cvat_dir / img_name).relative_to(cvat_dir)))
            train_file.write_text("\n".join(train_lines) + "\n")

            frame_idx += 1

    reader.close()
    print(f"Saved {frame_idx} samples to {out_dir}")
    create_cvat_annotation_zip(cvat_dir, out_dir)


# --------------------------
# CLI
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--metadata", required=False)
    parser.add_argument("--output", required=True)
    parser.add_argument("--topic-image", default="/camera")
    parser.add_argument("--topic-odom", default="/fmu/out/vehicle_odometry")
    parser.add_argument("--linear-thresh", type=float, default=DEFAULT_LINEAR_VEL_THRESH)
    parser.add_argument("--angular-thresh", type=float, default=DEFAULT_ANGULAR_VEL_THRESH)
    parser.add_argument("--frame_stride", type=int, default=DEFAULT_SAMPLE_STRIDE)
    parser.add_argument("--no-json", dest="save_json", action="store_false")

    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)

    process_with_rosbags(
        args.bag,
        args.metadata,
        args.output,
        args.topic_image,
        args.topic_odom,
        args.linear_thresh,
        args.angular_thresh,
        args.frame_stride,
        save_json=args.save_json,
    )


if __name__ == "__main__":
    main()
