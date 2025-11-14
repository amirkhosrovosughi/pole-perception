#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from px4_msgs.msg import VehicleOdometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from scripts.label_from_odometry import compute_bbox_from_odom, load_metadata, is_metadata_loaded

import numpy as np

class BBoxFromOdomNode(Node):
    """
    ROS 2 node for validating the pole detection data collection pipeline.

    The goal of this node is to validate the data collection pipeline for pole detection
    using the YOLO algorithm. It subscribes to the `/fmu/out/vehicle_odometry` topic, 
    computes the expected bounding box of the pole in the image frame based on the 
    vehicle's odometry and camera metadata, and outputs the result for verification 
    and debugging purposes.
    """
    def __init__(self):
        super().__init__('bbox_from_odom_node')

        # Hardcoded metadata path (you can later make this a ROS param)
        metadata_path = 'bags/2025_11_06-07_23_41/metadata.xml'


        self.get_logger().info(f"Loading metadata from: {metadata_path}")
        load_metadata(metadata_path)
        if not is_metadata_loaded():
            self.get_logger().error("Failed to load metadata!")
            return

        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        # Subscribe to odometry topic
        self.subscription = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odom_callback,
            qos_profile
        )

        image_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        self.image_sub = self.create_subscription(
            Image,
            '/camera',        # ← adjust if your topic is different
            self.image_callback,
            image_qos
        )
        self.subscription  # prevent unused variable warning
        self.last_bbox = None
        self.last_bbox_ts = None
        self.last_bbox_valid = False
        self.bridge = CvBridge()

        print("loaded ....")

        self.get_logger().info("BBoxFromOdomNode initialized and listening to /fmu/out/vehicle_odometry")

    def odom_callback(self, msg: VehicleOdometry):
        print("get an odom message")
        # Extract position and orientation
        pos_odom = np.array([msg.position[0], msg.position[1], msg.position[2]])
        q_odom = np.array([msg.q[0], msg.q[1], msg.q[2], msg.q[3]])  # (w, x, y, z)

        # Assume a dummy image size for now (e.g., 640x480)
        img_shape = (480, 640, 3)

        result = compute_bbox_from_odom(pos_odom, q_odom, img_shape, debug=False)

        if result["valid"]:
            bbox = result["bbox_norm"]
            self.get_logger().info(f"✅ BBox: {bbox}")
            self.last_bbox = result["bbox_norm"]
            self.last_bbox_valid = True
        else:
            reason = result.get("reason", "unknown")
            self.get_logger().warn(f"⚠️ Invalid bbox, reason: {reason}")
            self.last_bbox_valid = False

    def image_callback(self, msg):
        print("get an image message")
        # Convert ROS image → OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = frame.shape[:2]

        if self.last_bbox_valid and self.last_bbox is not None:
            x_c, y_c, bw, bh = self.last_bbox

            # Convert from normalized coords → pixel coords
            x_center = int(x_c * w)
            y_center = int(y_c * h)
            box_w = int(bw * w)
            box_h = int(bh * h)

            x1 = x_center - box_w // 2
            y1 = y_center - box_h // 2
            x2 = x_center + box_w // 2
            y2 = y_center + box_h // 2

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display window
        cv2.imshow("Pole Detection Debug", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = BBoxFromOdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
