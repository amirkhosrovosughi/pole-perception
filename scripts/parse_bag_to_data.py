#!/usr/bin/env python3
"""
parse_bag_to_yolo.py

Convert a ROS2 bag folder into a YOLOv8-ready dataset (images + labels).
Also produces a JSON file per image containing the 3D pole position in camera frame.

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
import xml.etree.ElementTree as ET
from math import sqrt
import numpy as np
import cv2
from tqdm import tqdm

# For quaternion / rotation math
from scipy.spatial.transform import Rotation as R

# Attempt to import bag readers. Preferred: rosbags. Fallback: rosbag2_py (ROS2).
_HAS_ROSBAGS = False
try:
    # 'rosbags' (pip install rosbags)
    from rosbags.rosbag2 import Reader as RosbagsReader
    from rosbags.typesys import deserialize_cdr
    _HAS_ROSBAGS = True
except Exception:
    _HAS_ROSBAGS = False

try:
    # ros2 built-in rosbag reader (requires ROS2 python packages sourced)
    import rosbag2_py
except Exception:
    rosbag2_py = None


# --------------------------
# Configurable defaults
# --------------------------
DEFAULT_LINEAR_VEL_THRESH = 0.2   # m/s (filter)
DEFAULT_ANGULAR_VEL_THRESH = 0.05 # rad/s (filter)
DEFAULT_POLE_HEIGHT = 1.0         # meters (used to compute bbox top/bottom)
CLASS_ID = 0                      # single-class (pole) for YOLO

# --------------------------
# Utility math helpers
# --------------------------
def ned_to_enu_vector(ned):
    """Convert a 3-vector from NED to ENU.
    As discussed: ENU = [y_ned, x_ned, -z_ned]
    """
    x_ned, y_ned, z_ned = ned
    return np.array([y_ned, x_ned, -z_ned], dtype=float)

def ned_quat_to_enu_quat(ned_quat_wxyz):
    """
    Convert quaternion (w, x, y, z) from NED to ENU using the rotation described.
    We'll do this by converting to rotation matrix, apply frame transforms, then back.
    """
    w, x, y, z = ned_quat_wxyz
    q = R.from_quat([x, y, z, w])  # scipy uses [x,y,z,w]
    rot_ned = q.as_matrix()

    # Rotation matrices from your C++:
    R_ned_to_enu = np.array([[0, 1, 0],
                             [1, 0, 0],
                             [0, 0, -1]], dtype=float)
    R_body = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=float)

    rot_enu = R_ned_to_enu.T @ rot_ned @ R_body
    r_enu = R.from_matrix(rot_enu)
    x_e, y_e, z_e, w_e = r_enu.as_quat()  # returns [x,y,z,w]
    return np.array([w_e, x_e, y_e, z_e], dtype=float)

def pose_to_transform(pose6):
    """Pose as [x,y,z, roll, pitch, yaw] -> homogeneous transform 4x4."""
    x, y, z, roll, pitch, yaw = pose6
    rot = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    T = np.eye(4)
    T[0:3, 0:3] = rot
    T[0:3, 3] = [x, y, z]
    return T

def transform_point(T, p):
    """Apply 4x4 transform T to 3-vector p, return 3-vector."""
    p4 = np.array([p[0], p[1], p[2], 1.0], dtype=float)
    r = T @ p4
    return r[0:3]

def invert_transform(T):
    Rm = T[0:3,0:3]
    tm = T[0:3,3]
    inv = np.eye(4)
    inv[0:3,0:3] = Rm.T
    inv[0:3,3] = -Rm.T @ tm
    return inv

def project_point_to_image(pt_cam, fx, fy, cx, cy):
    """Project 3D camera-frame point to pixel (u,v). Returns None if z<=0."""
    x, y, z = pt_cam
    if z <= 0:
        return None
    u = (fx * x) / z + cx
    v = (fy * y) / z + cy
    return float(u), float(v), float(z)

# --------------------------
# Parsing metadata.xml
# --------------------------
def parse_metadata(metadata_path):
    root = ET.parse(metadata_path).getroot()
    def parse_pose_text(tag):
        el = root.find(tag)
        if el is None:
            return None
        nums = [float(x) for x in el.text.strip().split()]
        if len(nums) != 6:
            raise ValueError(f"{tag} must contain 6 numbers (x y z r p y)")
        return nums

    base_to_camera = parse_pose_text('base_to_camera')
    pole_position = parse_pose_text('pole_position')
    cam_intr = root.find('camera_intrinsics')
    if cam_intr is None:
        raise ValueError("camera_intrinsics tag missing in metadata.xml (fx fy cx cy)")
    fx, fy, cx, cy = [float(x) for x in cam_intr.text.strip().split()]
    pole_height_el = root.find('pole_height')
    pole_height = float(pole_height_el.text.strip()) if pole_height_el is not None else DEFAULT_POLE_HEIGHT

    return {
        'base_to_camera': np.array(base_to_camera, dtype=float),
        'pole_position': np.array(pole_position, dtype=float),
        'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
        'pole_height': pole_height
    }

# --------------------------
# YOLO label writer
# --------------------------
def write_yolo_label(label_path, class_id, bbox_norm):
    """
    bbox_norm = (x_center_norm, y_center_norm, w_norm, h_norm)
    """
    with open(label_path, 'w') as f:
        x_c, y_c, w, h = bbox_norm
        f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

# --------------------------
# Main processing logic
# --------------------------
def process_with_rosbags(bag_dir, metadata, out_dir, topic_image, topic_odom,
                         lin_thresh, ang_thresh, pole_height, save_json=True):
    """
    Use 'rosbags' Reader (pip install rosbags) to iterate messages.
    This function expects rosbags to be installed. If not available, higher-level fallback is used.
    """
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Type, deserialize_cdr

    # Create output folders
    images_dir = Path(out_dir) / "images"
    labels_dir = Path(out_dir) / "labels"
    json_dir = Path(out_dir) / "meta3d"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    if save_json:
        json_dir.mkdir(parents=True, exist_ok=True)

    # Prepare transforms from metadata
    base_to_camera_pose = metadata['base_to_camera']    # x y z r p y (assumed ENU already)
    base_to_camera_T = pose_to_transform(base_to_camera_pose)
    pole_pose_world = metadata['pole_position']         # x y z r p y (world)
    pole_T_world = pose_to_transform(pole_pose_world)
    fx = metadata['fx']; fy = metadata['fy']; cx = metadata['cx']; cy = metadata['cy']

    # Open bag reader
    print("Opening bag (rosbags)...")
    reader = Reader(bag_dir)

    # Map topic -> connection info for quick checks
    conns = {c.topic: c for c in reader.connections}
    if topic_image not in conns or topic_odom not in conns:
        print("Available topics in bag:", [c.topic for c in reader.connections])
        raise RuntimeError("Requested topics not found in bag. Check topic names.")

    # We'll iterate all messages and keep latest odom; when we see an image, use latest odom that is close in time.
    latest_odom = None
    frame_idx = 0

    for connection, timestamp, rawdata in tqdm(reader.messages(), desc="Reading bag"):
        topic = connection.topic
        # Deserialize message
        msg = deserialize_cdr(rawdata, connection.msgtype)

        if topic == topic_odom:
            # px4_msgs/msg/VehicleOdometry fields (access depends on rosbags decoding)
            # We'll attempt to read position, q, velocity, angular_velocity
            try:
                pos = np.array([msg.position[0], msg.position[1], msg.position[2]], dtype=float)
                q = np.array([msg.q[0], msg.q[1], msg.q[2], msg.q[3]], dtype=float)  # w x y z?
                # Many times px4 msgs order is [w,x,y,z], but check and adapt if needed
                vel = np.array([msg.velocity[0], msg.velocity[1], msg.velocity[2]], dtype=float)
                ang_vel = np.array([msg.angular_velocity[0], msg.angular_velocity[1], msg.angular_velocity[2]], dtype=float)
            except Exception as e:
                # Different message field names — try alternatives or skip
                continue

            # Convert NED -> ENU for pos & quaternion & velocities
            pos_enu = ned_to_enu_vector(pos)
            q_enu = ned_quat_to_enu_quat(q)   # returns (w,x,y,z)
            # Build world (ENU) transform for base (vehicle)
            base_T_world = pose_to_transform([pos_enu[0], pos_enu[1], pos_enu[2], 0,0,0])
            # For orientation we incorporate quaternion rotation into base_T_world's rotation
            rot_enu = R.from_quat([q_enu[1], q_enu[2], q_enu[3], q_enu[0]]).as_matrix()
            base_T_world[0:3,0:3] = rot_enu

            latest_odom = {
                'T_world_base': base_T_world,
                'vel_lin': np.linalg.norm(vel),
                'vel_ang': np.linalg.norm(ang_vel)
            }

        elif topic == topic_image:
            if latest_odom is None:
                continue
            # Filter by thresholds
            if latest_odom['vel_lin'] > lin_thresh or latest_odom['vel_ang'] > ang_thresh:
                continue

            # Deserialize image data (we expect sensor_msgs/Image)
            # rosbags' deserialization for sensor_msgs.image fields provides .data (bytes) and encoding.
            try:
                height = int(msg.height)
                width = int(msg.width)
                encoding = msg.encoding  # e.g. 'rgb8' or 'bgr8' or 'mono8'
                arr = np.frombuffer(msg.data, dtype=np.uint8)
                # For typical encodings, we need to reshape:
                if encoding in ('rgb8', 'bgr8'):
                    img = arr.reshape((height, width, 3))
                    if encoding == 'rgb8':
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif encoding == 'mono8':
                    img = arr.reshape((height, width))
                else:
                    # try decoding with OpenCV as fallback
                    img = arr.reshape((height, width, -1))
            except Exception as e:
                # If deserialization did not yield expected fields, skip
                continue

            # Compute camera world transform: camera_T_world = base_T_world * base_to_camera
            base_T_world = latest_odom['T_world_base']
            cam_T_world = base_T_world @ base_to_camera_T  # transforms camera from camera->world? careful with convention
            # We want transform from world -> camera: T_cam_world = inv(cam_T_world)
            T_world_cam = invert_transform(cam_T_world)  # world -> camera transform

            # Get pole position in camera frame
            pole_world_pos = transform_point(pole_T_world, np.array([0.0, 0.0, 0.0]))  # pole origin in world
            pole_cam = transform_point(T_world_cam, pole_world_pos)  # in camera coords

            # For bbox vertical extents, compute pole top and bottom points in world and transform
            pole_bottom_world = pole_world_pos.copy()
            pole_top_world = pole_world_pos.copy()
            pole_top_world[2] += pole_height  # simple vertical offset in world frame
            pole_bottom_cam = transform_point(T_world_cam, pole_bottom_world)
            pole_top_cam = transform_point(T_world_cam, pole_top_world)

            # Project top & bottom to image
            proj_bottom = project_point_to_image(pole_bottom_cam, fx, fy, cx, cy)
            proj_top = project_point_to_image(pole_top_cam, fx, fy, cx, cy)
            proj_center = project_point_to_image(pole_cam, fx, fy, cx, cy)

            if proj_center is None or proj_bottom is None or proj_top is None:
                # pole not in front of camera
                continue

            u_c, v_c, zc = proj_center
            u_b, v_b, zb = proj_bottom
            u_t, v_t, zt = proj_top

            # bbox: center = (u_c, (v_b+v_t)/2), width small (we can estimate by projection of pole radius),
            # height = abs(v_b - v_t)
            bbox_h_px = abs(v_b - v_t)
            if bbox_h_px < 6:  # too small to be useful
                continue

            # width: choose a conservative small width relative to height
            bbox_w_px = bbox_h_px * 0.2

            img_h, img_w = img.shape[0], img.shape[1]
            x_center = u_c
            y_center = (v_b + v_t) / 2.0
            x_min = x_center - bbox_w_px / 2.0
            y_min = y_center - bbox_h_px / 2.0

            # Normalize for YOLO: x_center/img_w, y_center/img_h, w/img_w, h/img_h
            x_c_norm = x_center / img_w
            y_c_norm = y_center / img_h
            w_norm = bbox_w_px / img_w
            h_norm = bbox_h_px / img_h

            # Save image + label + optional JSON (3D cam coords)
            img_filename = f"{frame_idx:06d}.jpg"
            label_filename = f"{frame_idx:06d}.txt"
            json_filename = f"{frame_idx:06d}.json"

            cv2.imwrite(str(images_dir / img_filename), img)
            write_yolo_label(str(labels_dir / label_filename), CLASS_ID, (x_c_norm, y_c_norm, w_norm, h_norm))

            if save_json:
                meta = {
                    'frame_index': frame_idx,
                    'img_file': img_filename,
                    'pole_cam_xyz': pole_cam.tolist(),
                    'pole_bottom_cam_xyz': pole_bottom_cam.tolist(),
                    'pole_top_cam_xyz': pole_top_cam.tolist(),
                    'camera_fx_fy_cx_cy': [fx, fy, cx, cy],
                    'odom_linear_speed': float(latest_odom['vel_lin']),
                    'odom_angular_speed': float(latest_odom['vel_ang']),
                }
                with open(str(json_dir / json_filename), 'w') as jf:
                    json.dump(meta, jf, indent=2)

            frame_idx += 1

    print(f"Saved {frame_idx} samples to {out_dir}")
    reader.close()


# --------------------------
# Fallback reader (rosbag2_py)
# --------------------------
def process_with_rosbag2_py(bag_dir, metadata, out_dir, topic_image, topic_odom,
                            lin_thresh, ang_thresh, pole_height, save_json=True):
    """
    Fallback approach using rosbag2_py. NOTE: this path requires ROS2 python environment
    and a custom deserialization of messages (not fully portable).
    This function is a stub and may need adaptation to your ROS2 installation.
    """
    raise RuntimeError("rosbag2_py fallback not implemented in this script. Install 'rosbags' (pip install rosbags) for best results.")


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
    parser.add_argument("--pole-height", type=float, default=DEFAULT_POLE_HEIGHT)
    parser.add_argument("--no-json", dest="save_json", action="store_false")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = parse_metadata(args.metadata)

    if _HAS_ROSBAGS:
        process_with_rosbags(args.bag, metadata, out_dir, args.topic_image, args.topic_odom,
                             args.linear_thresh, args.angular_thresh, args.pole_height, save_json=args.save_json)
    else:
        print("rosbags package not available. Please install it: pip install rosbags")
        sys.exit(1)


if __name__ == "__main__":
    main()
