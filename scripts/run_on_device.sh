#!/bin/sh
# Chay tren device (qcm6490). Set duong dan thu vien QNN + HTP V68 skel.
QNN=/opt/qcom/qairt-latest/lib
export LD_LIBRARY_PATH=$QNN:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH=$QNN/hexagon-v68/unsigned
cd /opt/face_det_app
chmod +x ./face_det_lite
./face_det_lite ./face_det_lite.dlc ./images.jpg ./result.png "$@"
