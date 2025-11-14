# pole-perception-yolo

Detect and classify utility poles from drone imagery using YOLO-based object detection models.

## 📁 Project Structure
- `data/` — raw and labeled images (not tracked)
- `notebooks/` — Jupyter notebooks for training and validation
- `models/` — saved YOLO weights
- `assets/` — figures, demos, and visuals for documentation

## ⚙️ Quick Start
```bash
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


## Tool to validate data
This tool we subscribe to odom and image and show bounding box of the image, better to run in a seperate environment as it needs different depedencies version.
source ros_venv/bin/activate
python src/bbox_from_odom_node.py 
