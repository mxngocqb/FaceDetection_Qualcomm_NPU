# Camera → NPU → Web: Object Detection trực tiếp trên RB3 Gen2 (C thuần)

App C đọc **camera** → chạy detector trên **NPU Hexagon V68 (QNN HTP)** → phát kết quả
qua **web HTTP (MJPEG stream)**. Hiện chạy model **face_det_lite** (phát hiện khuôn mặt);
được thiết kế module-hoá để **đổi sang YOLOv8** chỉ bằng 2 hàm.

> Trạng thái đã kiểm chứng trên thiết bị: **~29 FPS**, stream MJPEG + JSON hoạt động.

```
File: cam_detect_web.c   (build ra: cam_detect_web)
```

---

## 1. Kiến trúc

```
 ┌─────────────┐   RGB thô    ┌──────────────────────────────────────────┐
 │ gst-launch  │   qua FIFO   │  cam_detect_web (C, đa luồng)              │
 │ qtiqmmfsrc  ├─────────────►│  ┌─ capture_thread ───────────────────┐   │
 │ (camera 0)  │ /tmp/cam_fifo│  │ đọc frame → preprocess → NPU(HTP)  │   │
 └─────────────┘              │  │ → postprocess → vẽ box → JPEG      │   │
                              │  └───────────┬─────────────────────────┘  │
                              │     frame chia sẻ (mutex) ▼                │
                              │  ┌─ HTTP server (port 8080) ──────────┐    │
   Trình duyệt ◄──────────────┼──┤ /  /stream(MJPEG)  /snapshot  /dets│    │
   http://IP:8080/            │  └────────────────────────────────────┘    │
                              └──────────────────────────────────────────┘
```

- **Camera**: dùng GStreamer `qtiqmmfsrc` (camera CSI qua Qualcomm camera stack), xuất RGB
  640×480 ghi vào **FIFO**; C đọc frame thô từ FIFO (không cần OpenCV / không link gstreamer).
- **NPU**: tái dùng scaffolding QNN HTP (nạp DLC, `graphExecute`) — chạy thật trên Hexagon V68.
- **Web**: HTTP server tự viết bằng socket + pthread.

---

## 2. Các endpoint HTTP

| Đường dẫn   | Nội dung |
|-------------|----------|
| `/`         | Trang HTML: video trực tiếp + danh sách detection + FPS |
| `/stream`   | MJPEG `multipart/x-mixed-replace` (video có vẽ box) |
| `/snapshot` | 1 ảnh JPEG mới nhất |
| `/dets`     | JSON: `{"fps":.., "dets":[{cls,score,x,y,r,b}, ...]}` |

---

## 3. Build (trên HOST)

```bash
cd face_det_app
TC=/tmp/aarch64--glibc--stable-2021.11-1
SDK=/đường/dẫn/qairt/2.43.0.260128
$TC/bin/aarch64-linux-gcc -O2 -std=gnu11 \
    -I$SDK/include/QNN -Ithird_party \
    cam_detect_web.c -o cam_detect_web \
    -ldl -lm -lpthread
```

## 4. Deploy

```bash
DEV=root@192.168.5.72        # pass: <device-password>
scp cam_detect_web run_cam_web.sh $DEV:/opt/face_det_app/
# DLC face đã có sẵn: /opt/face_det_app/face_det_lite.dlc
```

## 5. Chạy (trên DEVICE)

Quan trọng: **phải chạy detached** (setsid) để không bị SIGHUP khi đóng SSH:

```bash
ssh root@192.168.5.72
cd /opt/face_det_app
export LD_LIBRARY_PATH=/opt/qcom/qairt-latest/lib
export ADSP_LIBRARY_PATH=/opt/qcom/qairt-latest/lib/hexagon-v68/unsigned
export XDG_RUNTIME_DIR=/dev/socket/weston
export WAYLAND_DISPLAY=wayland-1
setsid ./cam_detect_web ./face_det_lite.dlc 8080 </dev/null >/tmp/cw.log 2>&1 &
```

Hoặc gọn:
```bash
ssh root@192.168.5.72 'cd /opt/face_det_app && setsid sh run_cam_web.sh </dev/null >/tmp/cw.log 2>&1 &'
```

**Xem kết quả**: mở trình duyệt máy cùng mạng → `http://192.168.5.72:8080/`

**Dừng**:
```bash
ssh root@192.168.5.72 'pkill -9 -f cam_detect_web; pkill -9 -f gst-launch'
```

Cú pháp: `cam_detect_web <model.dlc> [port=8080]`

---

## 6. Đổi sang model YOLOv8 (về sau)

Chỉ cần một **YOLOv8 QNN DLC** (vd `yolov8_det-qnn_dlc-w8a8` từ AI Hub) và sửa **2 hàm**
trong `cam_detect_web.c` (phần `// DETECTOR`):

1. **`detector_preprocess(rgb)`** — resize RGB camera → kích thước input YOLO (thường 640×640),
   sắp đúng layout/dtype mà DLC khai báo (in lúc khởi động). YOLOv8 w8a8 thường nhận **uint8**;
   layout NCHW `[1,3,640,640]` hoặc NHWC `[1,640,640,3]` (xem log `Graph ... inputs`).
2. **`detector_postprocess(out, maxd, thr)`** — decode đầu ra YOLOv8 `[1, 4+numClasses, 8400]`:
   - mỗi anchor: `cx,cy,w,h` + điểm số `numClasses` lớp; `score=max(lớp)`, `cls=argmax`.
   - box: `x=cx-w/2, y=cy-h/2, r=cx+w/2, b=cy+h/2` (scale từ 640 về kích thước hiển thị).
   - dequant uint8 bằng `scale/offset` đọc từ tensor; sau đó **NMS** (hàm `nms()` đã có).
3. Cập nhật **`class_name()`** đọc từ `labels.txt` (COCO 80 lớp).

Phần camera + NPU execute + HTTP **giữ nguyên**. `tool yolo_inspect` (kèm theo) in chính xác
shape/dtype/quant của DLC để bạn map cho đúng:
```bash
./yolo_inspect <model.dlc|model.bin>
```

---

## 7. Ghi chú & khắc phục sự cố

| Vấn đề | Xử lý |
|--------|-------|
| App tắt ngay khi đóng SSH | Phải chạy bằng `setsid ... </dev/null &` (đừng dùng `nohup` đơn thuần) |
| `/dets` báo fps=0, không frame | Camera chưa chạy — xem `/tmp/gst_cam.log`; đảm bảo không có tiến trình gst cũ giữ camera (`pkill gst-launch`) |
| Cảnh báo `GBM_ERR ... PRIME_FD` trong gst log | Vô hại (DRM buffer import warning); frame vẫn chảy ~30fps |
| Không mở được web | Kiểm tra IP (`ip -4 addr`), firewall, đúng cổng 8080; thiết bị & máy xem cùng mạng |
| Đổi độ phân giải | Sửa `CAP_W/CAP_H` (phải khớp input model nếu bỏ bước resize) |
| Camera khác | Đổi `camera=0` trong chuỗi gst (hàm `start_camera`) |

## 8. Hiệu năng (đã đo)

- Camera 1280×720 → scale 640×480, ~30 fps.
- face_det_lite trên HTP: `graphExecute` ~2–3 ms/frame.
- Tổng pipeline (đọc + NPU + vẽ + JPEG + phục vụ): **~29 FPS**.
