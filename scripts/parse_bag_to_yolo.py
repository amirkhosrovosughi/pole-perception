#!/usr/bin/env python3
"""
parse_bag_to_yolo.py

Reads a ros2 bag (using rosbags), extracts images and odometry and uses label_from_odometry
to compute YOLO-format bounding boxes and save images + labels + JSON metadata.

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

# local module
from label_from_odometry import load_metadata, compute_bbox_from_odom, is_metadata_loaded

# For quaternion / rotation math (used for converting px4 odometry -> transform)
from scipy.spatial.transform import Rotation as R

# Attempt to import bag readers. Preferred: rosbags. Fallback: rosbag2_py (ROS2).
_HAS_ROSBAGS = False

try:
    # Modern 'rosbags' API (v0.9+)
    from rosbags.highlevel import AnyReader
    _HAS_ROSBAGS = True
except ImportError:
    try:
        # Legacy API (pre-0.9)
        from rosbags.rosbag2 import Reader as RosbagsReader
        from rosbags.typesys import deserialize_cdr
        _HAS_ROSBAGS = True
    except ImportError:
        pass

try:
    import rosbag2_py
except ImportError:
    rosbag2_py = None

if not _HAS_ROSBAGS and rosbag2_py is None:
    print("❌ Neither 'rosbags' nor 'rosbag2_py' is available. Please install with: pip install rosbags")
    sys.exit(1)


# --------------------------
# Configurable defaults
# --------------------------
DEFAULT_LINEAR_VEL_THRESH = 0.5   # m/s (filter)
DEFAULT_ANGULAR_VEL_THRESH = 0.1  # rad/s (filter)
CLASS_ID = 0                      # single-class (pole) for YOLO


def create_cvat_annotation_zip(cvat_dir: Path, out_dir: Path) -> Path:
    """
    Creates a CVAT-compatible annotation.zip file.
    
    Args:
        cvat_dir (Path): Path to the directory containing the converted CVAT dataset.
        out_dir (Path): Directory where annotation.zip should be saved.

    Returns:
        Path: The path to the generated annotation.zip file.
    """
    import zipfile

    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "annotation.zip"

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file in cvat_dir.rglob('*'):
            zf.write(file, file.relative_to(cvat_dir.parent))

    print(f"CVAT annotation zip created: {zip_path}")
    return zip_path

def process_with_rosbags(bag_dir, metadata_path, out_dir, topic_image, topic_odom,
                         lin_thresh, ang_thresh, save_json=True):
    """
    Use 'rosbags' Reader to iterate messages and produce dataset.
    """
    from rosbags.rosbag2 import Reader
    from rosbags.serde import deserialize_cdr
    from rosbags.typesys import get_types_from_msg, register_types

    # try to register px4 msg types if available under ./msg
    px4_msgs_path = Path('msg')
    if px4_msgs_path.exists():
        msg_files = list(px4_msgs_path.glob('*.msg'))
        all_types = {}
        for msg_file in msg_files:
            package_name = 'px4_msgs'
            msg_name = msg_file.stem
            text = msg_file.read_text()
            types = get_types_from_msg(text, f'{package_name}/msg/{msg_name}')
            all_types.update(types)
        register_types(all_types)

    # Create output folders
    images_dir = Path(out_dir) / "images"
    labels_dir = Path(out_dir) / "labels"
    json_dir = Path(out_dir) / "meta3d"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    if save_json:
        json_dir.mkdir(parents=True, exist_ok=True)

    # Create output folder for CVAT
    cvat_dir = Path(out_dir) / "annotation"
    images_cvat_dir = cvat_dir / "obj_train_data" / "images"
    images_cvat_dir.mkdir(parents=True, exist_ok=True)

    # Write class names and obj.data
    class_names_file = cvat_dir / "obj.names"
    obj_data_file = cvat_dir / "obj.data"
    train_file = cvat_dir / "train.txt"

    # Single-class example
    with open(class_names_file, 'w') as f:
        f.write("pole\n")

    with open(obj_data_file, 'w') as f:
        f.write(f"classes = 1\n")
        f.write(f"train = {train_file.relative_to(cvat_dir)}\n")
        f.write(f"names = {class_names_file.relative_to(cvat_dir)}\n")
        f.write(f"backup = backup/\n")

    # If train.txt exists, load existing lines
    if train_file.exists():
        with open(train_file, 'r') as f:
            train_lines = f.read().splitlines()
    else:
        train_lines = []

    # Pre-load metadata
    load_metadata(metadata_path)
    if not is_metadata_loaded():
        raise RuntimeError("Failed to load metadata")

    # Open bag reader
    print("Opening bag (rosbags)...")
    reader = Reader(bag_dir)
    reader.open()

    # Map topic -> connection info for quick checks
    conns = {c.topic: c for c in reader.connections}
    if topic_image not in conns or topic_odom not in conns:
        print("Available topics in bag:", [c.topic for c in reader.connections])
        raise RuntimeError("Requested topics not found in bag. Check topic names.")

    latest_odom = None
    existing_images = list(images_cvat_dir.glob("*.jpg"))
    if existing_images:
        # get the max existing frame index
        last_idx = max(int(p.stem) for p in existing_images) + 1
    else:
        last_idx = 0
    frame_idx = last_idx

    for connection, timestamp, rawdata in tqdm(reader.messages(), desc="Reading bag"):
        topic = connection.topic
        msg = deserialize_cdr(rawdata, connection.msgtype)

        if topic == topic_odom:
            # Attempt to read px4 VehicleOdometry fields
            try:
                pos_odom = np.array([msg.position[0], msg.position[1], msg.position[2]], dtype=float)
                q_odom = np.array([msg.q[0], msg.q[1], msg.q[2], msg.q[3]], dtype=float)  # w x y z?
                vel = np.array([msg.velocity[0], msg.velocity[1], msg.velocity[2]], dtype=float)
                ang_vel = np.array([msg.angular_velocity[0], msg.angular_velocity[1], msg.angular_velocity[2]], dtype=float)
            except Exception:
                # field mismatch: skip this message
                continue

            latest_odom = {
                'pos_odom': pos_odom,
                'q_odom': q_odom,
                'vel_lin': float(np.linalg.norm(vel)),
                'vel_ang': float(np.linalg.norm(ang_vel))
            }

        elif topic == topic_image:
            if latest_odom is None:
                continue

            # Filter by thresholds
            if latest_odom['vel_lin'] > lin_thresh or latest_odom['vel_ang'] > ang_thresh:
                continue

            # Deserialize image
            try:
                height = int(msg.height)
                width = int(msg.width)
                encoding = msg.encoding
                arr = np.frombuffer(msg.data, dtype=np.uint8)
                if encoding in ('rgb8', 'bgr8'):
                    img = arr.reshape((height, width, 3))
                    if encoding == 'rgb8':
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif encoding == 'mono8':
                    img = arr.reshape((height, width))
                else:
                    img = arr.reshape((height, width, -1))
            except Exception:
                # skip malformed image message
                continue

            # Compute bbox via label_from_odometry
            pos_odom = latest_odom['pos_odom']
            q_odom = latest_odom['q_odom']
            res = compute_bbox_from_odom(pos_odom, q_odom, img.shape, pole_height=None, debug=False)

            if not res.get('valid', False):
                # optional: log reason when debugging
                print("skipping frame:", res.get('reason'))
                continue

            # Unpack results and save
            bbox_norm = res['bbox_norm']
            proj_center = res['proj_center']
            pole_cam = res['pole_cam']
            pole_bottom_cam = res['pole_bottom_cam']
            pole_top_cam = res['pole_top_cam']

            img_filename = f"{frame_idx:06d}.jpg"
            label_filename = f"{frame_idx:06d}.txt"
            json_filename = f"{frame_idx:06d}.json"

            cv2.imwrite(str(images_dir / img_filename), img)
            # write YOLO label
            with open(str(labels_dir / label_filename), 'w') as f:
                x_c, y_c, w, h = bbox_norm
                f.write(f"{CLASS_ID} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

            if save_json:
                meta = {
                    'frame_index': frame_idx,
                    'img_file': img_filename,
                    'pole_cam_xyz': pole_cam.tolist(),
                    'pole_bottom_cam_xyz': pole_bottom_cam.tolist(),
                    'pole_top_cam_xyz': pole_top_cam.tolist(),
                    'proj_center': proj_center,
                    'camera_fx_fy_cx_cy': [res.get('fx', None), res.get('fy', None), res.get('cx', None), res.get('cy', None)],
                    'odom_linear_speed': latest_odom['vel_lin'],
                    'odom_angular_speed': latest_odom['vel_ang'],
                }
                with open(str(json_dir / json_filename), 'w') as jf:
                    json.dump(meta, jf, indent=2)


            img_filename = f"{frame_idx:06d}.jpg"
            label_filename = f"{frame_idx:06d}.txt"

            # Save image
            cv2.imwrite(str(images_cvat_dir / img_filename), img)

            # Save label
            with open(str(images_cvat_dir / label_filename), 'w') as f:
                x_c, y_c, w, h = bbox_norm
                f.write(f"{CLASS_ID} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

            # Update train.txt with relative path to image
            train_lines.append(str((images_cvat_dir / img_filename).relative_to(cvat_dir)))

            # Write train.txt incrementally
            with open(train_file, 'w') as f:
                f.write("\n".join(train_lines) + "\n")

            frame_idx += 1

    print(f"Saved {frame_idx} samples to {out_dir}")
    zip_file = create_cvat_annotation_zip(cvat_dir, out_dir)
    reader.close()


# --------------------------
# CLI
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, help="Path to rosbag folder (directory used by ros2 bag).")
    parser.add_argument("--metadata", required=True, help="Path to metadata.xml (base_to_camera, pole_position, intrinsics).")
    parser.add_argument("--output", required=True, help="Output directory (e.g. data/train/)")
    parser.add_argument("--topic-image", default="/camera", help="Image topic name in bag")
    parser.add_argument("--topic-odom", default="/fmu/out/vehicle_odometry", help="Odometry topic name")
    parser.add_argument("--linear-thresh", type=float, default=DEFAULT_LINEAR_VEL_THRESH)
    parser.add_argument("--angular-thresh", type=float, default=DEFAULT_ANGULAR_VEL_THRESH)
    parser.add_argument("--no-json", dest="save_json", action="store_false")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # pass metadata path to loading routine (load once)
    load_metadata(args.metadata)

    if _HAS_ROSBAGS:
        process_with_rosbags(args.bag, args.metadata, out_dir, args.topic_image, args.topic_odom,
                             args.linear_thresh, args.angular_thresh, save_json=args.save_json)
    else:
        print("rosbags package not available. Please install it: pip install rosbags")
        sys.exit(1)


if __name__ == "__main__":
    main()
