# pole-perception-yolo

Detect and classify utility poles from drone imagery using YOLO-based object detection models.

## 📁 Project Structure
- `data/` — raw and labeled images (not tracked)
- `notebooks/` — Jupyter notebooks for training and validation
- `models/` — saved YOLO weights
- `assets/` — figures, demos, and visuals for documentation

## ⚙️ Quick Start

Collect data:
<pre>
ros2 bag record /camera /fmu/out/vehicle_odometry
</pre>

Need to provide a metadata.xml file for needed information to extract the label. It would be something like:
<pre>
<?xml version="1.0"?>
<scene>
  <!-- base -> camera pose as: x y z roll pitch yaw (radians) -->
  <base_to_camera>0.12 0.03 0.242 0 0.785 0</base_to_camera>

  <!-- pole position in world frame: x y z roll pitch yaw -->
  <!-- this should be world coordinates where you placed the pole in Gazebo -->
  <pole_position>2 0 -0.5 0 0 0</pole_position>

  <!-- camera intrinsics: fx fy cx cy (we'll parse K entry in your file) -->
  <camera_intrinsics>1393 1393 960 540</camera_intrinsics>

  <!-- pole physical height (meters) used to estimate bbox extent -->
  <pole_height>1.0</pole_height>
</scene>

</pre>

Setup the environment

<pre>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install rosbags==0.9.21

python scripts/parse_bag_to_yolo.py \
  --bag bags/2025_11_06-07_23_41/rosbag2_2025_11_06-07_23_41 \
  --metadata bags/2025_11_06-07_23_41/metadata.xml \
  --output data/train \
  --topic-image /camera \
  --topic-odom /fmu/out/vehicle_odometry
</pre>


## Tool to validate data
This tool we subscribe to odom and image and show bounding box of the image, better to run in a seperate environment as it needs different depedencies version.
<pre>
source ros_venv/bin/activate
python src/bbox_from_odom_node.py
</pre>
