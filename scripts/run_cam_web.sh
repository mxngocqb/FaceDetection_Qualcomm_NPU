#!/bin/sh
# Chay app camera+NPU+web tren device (qcm6490)
QNN=/opt/qcom/qairt-latest/lib
export LD_LIBRARY_PATH=$QNN:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH=$QNN/hexagon-v68/unsigned
export XDG_RUNTIME_DIR=/dev/socket/weston
export WAYLAND_DISPLAY=wayland-1
cd /opt/face_det_app
MODEL=${1:-./face_det_lite.dlc}
PORT=${2:-8080}
exec ./cam_detect_web "$MODEL" "$PORT"
