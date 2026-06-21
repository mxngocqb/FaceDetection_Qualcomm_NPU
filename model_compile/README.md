# model_compile — Tạo QNN model (w8a8) cho QCS6490 qua Qualcomm AI Hub

Vì các DLC/.bin YOLO cũ trên device **không tương thích** QNN runtime, và HTP **bắt buộc model
quantized (w8a8) với I/O cũng quantized**, ta tạo model mới bằng AI Hub (quantize + compile trên
cloud Qualcomm, miễn phí với API token).

## 1. Cài đặt (host, Python 3.x)
```bash
pip install -r requirements.txt
# Lấy token MIỄN PHÍ: https://aihub.qualcomm.com -> Settings -> API token
qai-hub configure --api_token <YOUR_FREE_TOKEN>
# (hoặc: export QAI_HUB_API_TOKEN=<YOUR_FREE_TOKEN>)
```
> Token là secret — KHÔNG commit vào repo.

## 2. Chuẩn bị ONNX
- ONNX YOLOv8 nên **nhúng sẵn hậu xử lý** (output: `boxes[1,8400,4]`, `scores[1,8400]`,
  `class_idx[1,8400]`) để postprocess phía C đơn giản.
- Input: `image [1,3,640,640]`, RGB NCHW, giá trị [0,1] (chuẩn hoá ngoài model).

## 3. Chụp ảnh calibration (trên DEVICE)
```bash
# tren device
sh calib_capture.sh 640 640 10
# tren host: keo ve
mkdir -p calib
scp 'root@192.168.5.72:/tmp/calib_*.rgb' ./calib/
```
Mỗi file `.rgb` = raw RGB `width*height*3` bytes (uint8). ~30 frame là đủ để chạy;
muốn chính xác hơn nên dùng nhiều ảnh **đa dạng** (vd ảnh COCO).

## 4. Quantize + Compile -> context binary
```bash
python3 compile_yolo_aihub.py \
    --onnx yolo.onnx --calib-dir ./calib \
    --width 640 --height 640 \
    --device "Dragonwing RB3 Gen 2 Vision Kit" \
    --out ../models/yolov8_w8a8.bin
```
Script sẽ: gộp external-data ONNX → upload → **quantize w8a8** → **compile**
`--target_runtime qnn_context_binary --quantize_io` → tải `.bin`.

## 5. Kiểm tra I/O của model
```bash
# build yolo_inspect (xem scripts/build.sh), copy len device roi:
./yolo_inspect ../models/yolov8_w8a8.bin
```

## Ghi chú
- `--quantize_io` là **bắt buộc**: HTP từ chối model có I/O float
  (lỗi `Tensor 'image' has a floating-point type ... not supported`).
- Có thể đổi `--runtime qnn_dlc` để ra DLC (nạp bằng `systemDlcCreateFromFile`),
  nhưng context binary được finalize sẵn → khởi động nhanh hơn trên device.
- Thiết bị mục tiêu trên AI Hub: **"Dragonwing RB3 Gen 2 Vision Kit"** (QCS6490, qc_linux).
