#!/bin/sh
# calib_capture.sh — chup anh calibration (raw RGB 640x640) tu camera tren DEVICE.
# Chay TREN DEVICE:  sh calib_capture.sh [width] [height] [seconds]
# Sau do tren HOST keo ve:  scp 'root@DEV:/tmp/calib_*.rgb' ./calib/
W=${1:-640}
H=${2:-640}
SEC=${3:-10}
export XDG_RUNTIME_DIR=/dev/socket/weston
export WAYLAND_DISPLAY=wayland-1
rm -f /tmp/calib_*.rgb
echo "Chup ${SEC}s -> /tmp/calib_%03d.rgb (RGB ${W}x${H})"
timeout "$SEC" gst-launch-1.0 qtiqmmfsrc camera=0 ! \
  video/x-raw,width=1280,height=720,format=NV12 ! videoconvert ! videoscale ! \
  video/x-raw,format=RGB,width="$W",height="$H" ! \
  multifilesink location=/tmp/calib_%03d.rgb
echo "Da chup $(ls /tmp/calib_*.rgb 2>/dev/null | wc -l) frame"
