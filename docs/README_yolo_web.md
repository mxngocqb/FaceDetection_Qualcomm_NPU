# YOLOv8 Object Detection: Camera → NPU → Web (C thuần) trên RB3 Gen2

App C đọc **camera** → chạy **YOLOv8 (w8a8)** trên **NPU Hexagon V68 (QNN HTP)** → phát kết quả
(video + box + nhãn lớp) qua **web HTTP (MJPEG)**.

> Đã kiểm chứng trên thiết bị: **~27 FPS**, phát hiện đúng vật thể (vd "orange") + vẽ box, web hoạt động.

```
Nguồn:  cam_yolo_web.c        ->  binary: cam_yolo_web
Model:  yolov8_w8a8.bin       (QNN context binary, w8a8, cho QCS6490)
Chạy:   run_yolo_web.sh
```

---

## 1. Mở xem
Trình duyệt (máy cùng mạng): **http://192.168.5.72:8080/**
Endpoints: `/` (trang HTML), `/stream` (MJPEG), `/snapshot` (1 ảnh JPEG), `/dets` (JSON).

## 2. Model đến từ đâu (đã làm)
Local YOLO DLC/.bin trên device **không tương thích** QNN runtime → đã tạo model mới qua **Qualcomm AI Hub**:
1. Lấy ONNX YOLOv8 (đã nhúng decode: out `boxes[1,8400,4]`, `scores[1,8400]`, `class_idx[1,8400]`).
2. `qai-hub` **quantize job** (w8a8, INT8) với 32 ảnh calibration chụp từ chính camera.
3. **compile job** → `--target_runtime qnn_context_binary --quantize_io` → `yolov8_w8a8.bin`.
   (Lưu ý: HTP **bắt buộc** I/O quantized — phải có `--quantize_io`; bản float bị từ chối.)

Script tái tạo: `/tmp/opencode/quantize_yolo.py` + `compile_io.py` (cần qai-hub token).

## 3. I/O model (đọc tự động lúc chạy)
| Tensor | Shape | Kiểu | Ghi chú |
|--------|-------|------|---------|
| `image` | `[1,3,640,640]` | ufixed8, scale 1/255 | NCHW RGB, q = giá trị pixel |
| `output_0` boxes | `[1,8400,4]` | ufixed8, scale 2.631 off −8 | xyxy, hệ toạ độ 640 |
| `output_1` scores | `[1,8400]` | ufixed8, scale 1/256 | đã sigmoid (0..1) |
| `output_2` class_idx | `[1,8400]` | uint8 | chỉ số lớp COCO (0..79) |

## 4. Pipeline C (trong `cam_yolo_web.c`)
- **Camera**: GStreamer `qtiqmmfsrc` → RGB 640×640 ghi FIFO → C đọc frame thô.
- **NPU**: nạp **context binary** bằng `systemContextGetBinaryInfo` + `contextCreateFromBinary`
  (đã finalize sẵn → khởi động tức thì, không compose trên device), rồi `graphExecute`.
- **Preprocess** (`detector_preprocess`): RGB interleaved → **uint8 NCHW** (q = pixel; scale 1/255).
- **Postprocess** (`detector_postprocess`): dequant `scores/boxes`, lọc theo ngưỡng (0.40),
  lấy lớp từ `class_idx`, **NMS theo lớp** (IoU 0.45). Box ở hệ 640 → vẽ thẳng lên ảnh 640×640.
- **Web**: HTTP server (socket + pthread), MJPEG + JSON.

## 5. Build / Deploy / Run

```bash
# BUILD (host)
TC=/tmp/aarch64--glibc--stable-2021.11-1 ; SDK=.../qairt/2.43.0.260128
$TC/bin/aarch64-linux-gcc -O2 -std=gnu11 -I$SDK/include/QNN -Ithird_party \
    cam_yolo_web.c -o cam_yolo_web -ldl -lm -lpthread

# DEPLOY
scp cam_yolo_web yolov8_w8a8.bin run_yolo_web.sh root@192.168.5.72:/opt/face_det_app/

# RUN (device) — phải detached
ssh root@192.168.5.72 'cd /opt/face_det_app && setsid sh run_yolo_web.sh </dev/null >/tmp/yolo.log 2>&1 &'
# Dừng
ssh root@192.168.5.72 'pkill -9 -f cam_yolo_web; pkill -9 -f gst-launch'
```

Cú pháp: `cam_yolo_web <model.bin> [port=8080]`

## 6. Tinh chỉnh
| Muốn | Sửa |
|------|-----|
| Nhiều/ít vật hơn | ngưỡng `0.40f` trong `capture_thread` |
| Độ gộp NMS | `nms(out,nd,0.45f)` trong `detector_postprocess` |
| Độ chính xác lượng tử tốt hơn | calibrate bằng nhiều ảnh đa dạng (COCO) thay vì 32 frame 1 cảnh |
| Đỡ méo ảnh | dùng letterbox 640×640 (giữ tỉ lệ) thay vì scale thẳng |
| Đổi camera | `camera=0` trong `start_camera()` |

## 7. Hiệu năng (đo)
- Context binary: khởi động ~tức thì (không compose/finalize trên device).
- YOLOv8 w8a8 trên HTP V68 + camera + JPEG + web: **~27 FPS**.

## 8. Ghi chú
- Calibration hiện chỉ 32 frame của 1 cảnh tĩnh → quantization hơi kém nhạy với lớp ít gặp.
  Để bắt nhiều lớp hơn, hạ ngưỡng hoặc quantize lại với bộ ảnh đa dạng.
- Bảo mật: token qai-hub đã dùng để build — **nên rotate** sau khi xong.
