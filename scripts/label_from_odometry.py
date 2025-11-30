#!/usr/bin/env python3
"""
label_from_odometry.py

Utilities to compute a 2D YOLO bounding box for a pole from a base->world transform.
This module:
 - loads & caches metadata once (load_metadata)
 - exposes compute_bbox_from_odom(T_world_base, img_shape, pole_height=None, debug=False)
   which returns a dict describing whether a bbox is valid and the normalized bbox coords.
"""

from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R

# cached metadata store
_metadata_cache = None

# defaults
DEFAULT_POLE_HEIGHT = 1.0
MIN_BBOX_HEIGHT_PX = 6  # same filter as main script (you can change when calling compute_bbox...)

_M_NED_TO_ENU = np.array([[0.0, 1.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 0.0, -1.0]], dtype=float)


# --------------------------
# Parsing metadata (internal)
# --------------------------
def _parse_metadata(metadata_path):
    root = ET.parse(str(metadata_path)).getroot()

    def parse_pose_text(tag):
        el = root.find(tag)
        if el is None:
            return None
        nums = [float(x) for x in el.text.strip().split()]
        if len(nums) != 6:
            raise ValueError(f"{tag} must contain 6 numbers (x y z roll pitch yaw)")
        return nums

    def parse_multiple_poses(parent_tag):
        parent = root.find(parent_tag)
        if parent is None:
            return []
        poses = []
        for pole_el in parent.findall('pole'):
            nums = [float(x) for x in pole_el.text.strip().split()]
            if len(nums) != 6:
                raise ValueError("Each <pole> must contain 6 numbers (x y z roll pitch yaw)")
            poses.append(nums)
        return poses

    base_to_camera = parse_pose_text('base_to_camera')
    poles_position = parse_multiple_poses('poles_position')

    cam_intr = root.find('camera_intrinsics')
    if cam_intr is None:
        raise ValueError("camera_intrinsics tag missing in metadata.xml (fx fy cx cy)")
    fx, fy, cx, cy = [float(x) for x in cam_intr.text.strip().split()]

    pole_height_el = root.find('pole_height')
    pole_height = float(pole_height_el.text.strip()) if pole_height_el is not None else DEFAULT_POLE_HEIGHT

    return {
        'base_to_camera_pose': np.array(base_to_camera, dtype=float),
        'poles_position_pose': np.array(poles_position, dtype=float),
        'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
        'pole_height': pole_height
    }


# --------------------------
# Geometry helpers
# --------------------------
def pose_to_transform(pose6):
    """Pose as [x,y,z, roll, pitch, yaw] -> homogeneous transform 4x4."""
    x, y, z, roll, pitch, yaw = pose6
    rot = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    T = np.eye(4, dtype=float)
    T[0:3, 0:3] = rot
    T[0:3, 3] = [x, y, z]
    return T

def poses_to_transforms(poses6):
    """
    Convert an Nx6 array/list of poses into a list of 4x4 transforms.
    Each pose is [x, y, z, roll, pitch, yaw].
    """
    transforms = []
    for pose6 in poses6:
        T = pose_to_transform(pose6)
        transforms.append(T)
    return np.array(transforms, dtype=float) 

def invert_transform(T):
    Rm = T[0:3, 0:3]
    tm = T[0:3, 3]
    inv = np.eye(4, dtype=float)
    inv[0:3, 0:3] = Rm.T
    inv[0:3, 3] = -Rm.T @ tm
    return inv


def transform_point(T, p):
    p4 = np.array([p[0], p[1], p[2], 1.0], dtype=float)
    r = T @ p4
    return r[0:3]


def camera_physical_to_optical_T():
    """
    Transform from simulator/PX4 camera physical frame -> OpenCV optical frame.
    This maps camera axes such that the optical forward axis becomes +Z (OpenCV convention).
    Adjust if your simulator uses different frame.
    """
    T = np.eye(4, dtype=float)
    # This rotation was chosen to map physical frame axes to optical (+Z forward)
    # R = [[0,0,1],[1,0,0],[0,1,0]] as used earlier.
    Rm = np.array([
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0]
    ], dtype=float)
    T[:3, :3] = Rm
    return T


def project_point_to_image(pt_cam, fx, fy, cx, cy, debug=False):
    """
    Project 3D camera-frame point (optical frame, where +Z forward) to pixel coords.
    Returns (u, v, z) or None if z <= 0.
    """
    x, y, z = float(pt_cam[0]), float(pt_cam[1]), float(pt_cam[2])
    if z <= 0:
        if debug:
            print(f"  skipping projection: z <= 0, point_cam = {pt_cam}")
        return None
    u = (fx * x) / z + cx
    v = (fy * y) / z + cy
    if debug:
        print(f"  projected pixel: ({u:.1f}, {v:.1f}), depth z={z:.3f}")
    return float(u), float(v), float(z)


# --------------------------
# Public API: load / check
# --------------------------
def load_metadata(metadata_path):
    """
    Parse metadata.xml and cache values (including precomputed transforms).
    Call this once at startup.
    """
    global _metadata_cache
    metadata_path = Path(metadata_path)
    raw = _parse_metadata(metadata_path)

    base_to_camera_T = pose_to_transform(raw['base_to_camera_pose'])  # base->camera physical

    # If Poles Position is in NED need to run this instead

    poles_T_world = poses_to_transforms(raw['poles_position_pose'])

    # store prepared items
    _metadata_cache = {
        'base_to_camera_T': base_to_camera_T,
        'poles_T_world': poles_T_world,
        'fx': raw['fx'],
        'fy': raw['fy'],
        'cx': raw['cx'],
        'cy': raw['cy'],
        'pole_height': raw['pole_height']
    }
    return _metadata_cache


def is_metadata_loaded():
    return _metadata_cache is not None

# --------------------------
# Helpers included here for odom->transform creation (kept minimal)
# --------------------------
def ned_to_enu_vector(ned):
    x_ned, y_ned, z_ned = ned
    return np.array([y_ned, x_ned, -z_ned], dtype=float)


def ned_quat_to_enu_quat(ned_quat_wxyz):
    # Convert (w,x,y,z) from NED to ENU quaternion (w,x,y,z)
    w, x, y, z = ned_quat_wxyz
    q = R.from_quat([x, y, z, w])  # scipy uses [x,y,z,w]
    rot_ned = q.as_matrix()
    R_ned_to_enu = np.array([[0, 1, 0],
                             [1, 0, 0],
                             [0, 0, -1]], dtype=float)
    R_body = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
    rot_enu = R_ned_to_enu.T @ rot_ned @ R_body
    r_enu = R.from_matrix(rot_enu)
    x_e, y_e, z_e, w_e = r_enu.as_quat()
    return np.array([w_e, x_e, y_e, z_e], dtype=float)


def pose_to_transform(pose6):
    x, y, z, roll, pitch, yaw = pose6
    rot = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    T = np.eye(4, dtype=float)
    T[0:3, 0:3] = rot
    T[0:3, 3] = [x, y, z]
    return T

def ned_to_enu_pose(pose6_ned):
    """
    Convert pose in NED convention [x, y, z, roll, pitch, yaw] (radians)
    to ENU convention [x_e, y_e, z_e, roll_e, pitch_e, yaw_e].
    """
    x, y, z, roll, pitch, yaw = pose6_ned
    # translation
    trans_ned = np.array([x, y, z], dtype=float)
    trans_enu = _M_NED_TO_ENU @ trans_ned

    # rotation: build R_ned from euler in NED axes (xyz = roll,pitch,yaw)
    R_ned = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    R_enu = _M_NED_TO_ENU @ R_ned @ _M_NED_TO_ENU.T
    roll_e, pitch_e, yaw_e = R.from_matrix(R_enu).as_euler('xyz')

    return np.array([trans_enu[0], trans_enu[1], trans_enu[2], roll_e, pitch_e, yaw_e], dtype=float)

# --------------------------
# compute_bbox_from_odom
# --------------------------
def compute_bbox_from_odom(pos_odom, q_odom, img_shape, pole_height=None, debug=False):
    """
    Compute the 2D image bounding box of a vertical pole based on vehicle odometry and
    camera extrinsic parameters.

    This function projects a known-height pole (in world coordinates) into the camera image
    using the vehicle’s odometry pose (position + orientation) and preloaded camera metadata.
    It outputs both the normalized bounding box and intermediate 3D projection data.

    Args:
        pos_odom (np.ndarray | list[float]): 
            The vehicle position in the NED (North-East-Down) odometry frame, 
            as [x, y, z] in meters.
        
        q_odom (np.ndarray | list[float]): 
            The vehicle orientation quaternion in NED frame, formatted as (w, x, y, z).
        
        img_shape (tuple[int]): 
            Shape of the image as (H, W, ...) — typically obtained from `img.shape`.

        pole_height (float, optional): 
            Optional override for the pole height in meters. 
            If not provided, the height from metadata is used.

        debug (bool, optional): 
            If True, prints intermediate transformation details for verification.

    Returns a dict:
      {
         "valid": bool,
         "bbox_norm": (x_c_norm, y_c_norm, w_norm, h_norm) or None,
         "proj_center": (u_c, v_c, zc) or None,
         "proj_top": ...,
         "proj_bottom": ...,
         "pole_cam": np.array(...),
         "pole_top_cam": np.array(...),
         "pole_bottom_cam": np.array(...),
      }

    Notes:
        - Requires metadata to be loaded beforehand via `load_metadata(path)`.
        - The function internally converts NED odometry pose to ENU coordinates
          to match the camera/world frame convention.
        - Uses transforms from metadata:
            - `base_to_camera_T`: vehicle base → physical camera
            - `camera_physical_to_optical_T`: physical → optical camera frame
        - Final camera transform is composed as:
            ```
            cam_T_world = T_world_base @ T_base_cam_physical @ T_cam_physical_to_optical
            ```
    """
    if _metadata_cache is None:
        if debug:
            print("[label_from_odometry] metadata not loaded. call load_metadata(path) first")
        return {"valid": False, "reason": "metadata_not_loaded"}

    md = _metadata_cache
    fx, fy, cx, cy = md['fx'], md['fy'], md['cx'], md['cy']
    pole_h = md['pole_height'] if pole_height is None else float(pole_height)

    # transforms from metadata
    T_base_cam_physical = md['base_to_camera_T']  # base -> physical camera
    T_cam_physical_to_optical = camera_physical_to_optical_T()


    pos_enu = ned_to_enu_vector(pos_odom)
    q_enu = ned_quat_to_enu_quat(q_odom)   # returns (w,x,y,z)
    # Build world (ENU) transform for base (vehicle)
    T_world_base = pose_to_transform([pos_enu[0], pos_enu[1], pos_enu[2], 0, 0, 0])
    rot_enu = R.from_quat([q_enu[1], q_enu[2], q_enu[3], q_enu[0]]).as_matrix()
    T_world_base[0:3, 0:3] = rot_enu

    # Compose camera in world: cam_T_world = base_to_world @ base_to_camera @ phys->optical
    # (This matches your earlier approach: cam_T_world = base_to_world @ base_to_camera_T @ correction)
    cam_T_world = T_world_base @ T_base_cam_physical @ T_cam_physical_to_optical
    T_world_cam = invert_transform(cam_T_world)

    if debug:
        print("[label_from_odometry] cam_T_world:\n", cam_T_world)

    # pole origin in world (pole_T_world is full pose, but we only need the translation)
    poles_T_world = md['poles_T_world']

    results = []

    for pole_T_world in poles_T_world:
        pole_world_pos = transform_point(pole_T_world, np.array([0.0, 0.0, 0.0]))
        pole_world_pos[2] += pole_h / 2
        pole_cam = transform_point(T_world_cam, pole_world_pos)

        # top/bottom in world and then in camera frame
        pole_bottom_world = pole_world_pos.copy()
        pole_top_world = pole_world_pos.copy()
        pole_bottom_world[2] -= pole_h / 2
        pole_top_world[2] += pole_h / 2  # vertical offset in world frame (z axis)
        pole_bottom_cam = transform_point(T_world_cam, pole_bottom_world)
        pole_top_cam = transform_point(T_world_cam, pole_top_world)

        if debug:
            print("  pole_world_pos:", pole_world_pos)
            print("  pole_cam:", pole_cam)
            print("  pole_bottom_cam:", pole_bottom_cam)
            print("  pole_top_cam:", pole_top_cam)

        # project
        proj_center = project_point_to_image(pole_cam, fx, fy, cx, cy, debug=debug)
        proj_bottom = project_point_to_image(pole_bottom_cam, fx, fy, cx, cy, debug=debug)
        proj_top = project_point_to_image(pole_top_cam, fx, fy, cx, cy, debug=debug)

        if proj_center is None or proj_bottom is None or proj_top is None:
            if debug:
                print("[label_from_odometry] one or more projections invalid -> not visible")
            result =  {"valid": False, "reason": "not_visible",
                    "proj_center": proj_center, "proj_bottom": proj_bottom, "proj_top": proj_top,
                    "pole_cam": pole_cam, "pole_top_cam": pole_top_cam, "pole_bottom_cam": pole_bottom_cam}
            results.append(result)
            continue

        # pixel coords
        u_c, v_c, zc = proj_center
        u_b, v_b, zb = proj_bottom
        u_t, v_t, zt = proj_top

        bbox_h_px = abs(v_b - v_t)
        # if bbox_h_px < MIN_BBOX_HEIGHT_PX: #TODO looks necessary, but commented temporaty for furthur verifications
        #     if debug:
        #         print(f"[label_from_odometry] bbox height too small: {bbox_h_px}px (threshold {MIN_BBOX_HEIGHT_PX})")
        #     result = {"valid": False, "reason": "too_small", "bbox_h_px": bbox_h_px,
        #             "proj_center": proj_center, "proj_bottom": proj_bottom, "proj_top": proj_top}
        #       results.append(result)
        #       continue

        # choose conservative width as fraction
        bbox_w_px = bbox_h_px * 0.4

        img_h = int(img_shape[0])
        img_w = int(img_shape[1])

        x_center = u_c
        y_center = (v_b + v_t) / 2.0

        # normalized YOLO-style (x_center/img_w, y_center/img_h, w/img_w, h/img_h)
        x_c_norm = x_center / img_w
        y_c_norm = y_center / img_h
        w_norm = bbox_w_px / img_w
        h_norm = bbox_h_px / img_h

        result = {
            "valid": True,
            "bbox_norm": (x_c_norm, y_c_norm, w_norm, h_norm),
            "bbox_px": (x_center, y_center, bbox_w_px, bbox_h_px),
            "proj_center": proj_center,
            "proj_top": proj_top,
            "proj_bottom": proj_bottom,
            "pole_cam": pole_cam,
            "pole_top_cam": pole_top_cam,
            "pole_bottom_cam": pole_bottom_cam,
            "cam_T_world": cam_T_world
        }

        results.append(result)

    print(f"Insider: results.len {len(results)}")
    return results
