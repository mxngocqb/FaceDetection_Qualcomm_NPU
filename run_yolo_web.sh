#!/bin/sh
# Chay YOLOv8 camera+NPU+web tren device (qcm6490)
QNN=/opt/qcom/qairt-latest/lib
export LD_LIBRARY_PATH=$QNN:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH=$QNN/hexagon-v68/unsigned
export XDG_RUNTIME_DIR=/dev/socket/weston
export WAYLAND_DISPLAY=wayland-1
cd /opt/face_det_app
MODEL=${1:-./yolov8_w8a8.bin}
PORT=${2:-8080}
exec ./cam_yolo_web "$MODEL" "$PORT"
