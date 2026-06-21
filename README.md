# Edge AI trên Qualcomm RB3 Gen 2 (QCS6490) — Inference bằng C trên NPU Hexagon V68

Bộ ứng dụng **C thuần** chạy AI trên **NPU Hexagon V68 (QNN HTP)** của **Qualcomm Robotics RB3 Gen 2 (QCS6490 / qcm6490)**:

| # | Ứng dụng | Mô tả | Model |
|---|----------|-------|-------|
| 1 | **face_det_lite** | Phát hiện khuôn mặt trên **1 ảnh** → ghi ảnh có vẽ box | FaceDetLite QNN DLC (w8a8) |
| 2 | **cam_detect_web** | **Camera → NPU → Web (MJPEG)**, phát hiện khuôn mặt realtime | FaceDetLite QNN DLC |
| 3 | **cam_yolo_web** | **Camera → NPU → Web (MJPEG)**, **YOLOv8** object detection 80 lớp COCO | YOLOv8 QNN context binary (w8a8) |

Tất cả inference chạy thật trên **NPU** (đã kiểm chứng: VTCM, FastRPC, nhanh ~16× so với CPU).

> Kết quả đã đo trên thiết bị: face_det_lite ~2–3 ms/ảnh; camera+web ~27–29 FPS.

---

## Mục lục
1. [Kiến trúc chung](#1-kiến-trúc-chung)
2. [Phần cứng & môi trường](#2-phần-cứng--môi-trường)
3. [Cấu trúc repo](#3-cấu-trúc-repo)
4. [Toolchain & Build](#4-toolchain--build)
5. [App 1 — Phát hiện khuôn mặt trên ảnh](#5-app-1--face_det_lite-ảnh-tĩnh)
6. [App 2 — Camera + khuôn mặt + Web](#6-app-2--cam_detect_web)
7. [App 3 — Camera + YOLOv8 + Web](#7-app-3--cam_yolo_web)
8. [Model lấy từ đâu](#8-model-lấy-từ-đâu)
9. [Xác nhận chạy trên NPU](#9-xác-nhận-chạy-trên-npu)
10. [Khắc phục sự cố](#10-khắc-phục-sự-cố)
11. [Bảo mật](#11-bảo-mật)

---

## 1. Kiến trúc chung

Mọi app dùng chung khung nạp model QNN qua `dlopen` (không link `libQnn*` lúc build → chỉ cần header + `-ldl -lm -lpthread`):

```
QnnInterface_getProviders        (libQnnHtp.so)      -> backend HTP V68
QnnSystemInterface_getProviders  (libQnnSystem.so)   -> system interface
backendCreate -> deviceCreate -> contextCreate
   ── DLC:   systemDlcCreateFromFile -> systemDlcComposeGraphs -> graphFinalize
   ── .bin:  systemContextGetBinaryInfo -> contextCreateFromBinary   (đã finalize sẵn)
graphRetrieve -> graphExecute        (chạy trên NPU)
```

App camera/web bổ sung: **GStreamer `qtiqmmfsrc`** xuất frame RGB qua **FIFO** → C đọc thô →
preprocess → `graphExecute` → postprocess → vẽ box → encode JPEG → **HTTP MJPEG server** (socket + pthread).

---

## 2. Phần cứng & môi trường

| Hạng mục | Giá trị |
|----------|---------|
| Thiết bị | Qualcomm **RB3 Gen 2** (Dragonwing), SoC **QCS6490** (`qcm6490`) |
| NPU | Hexagon **V68** HTP |
| OS | QCOM Reference Distro (Wayland), kernel 6.6.x, aarch64, **glibc 2.35** |
| QNN runtime (device) | `/opt/qcom/qairt-latest` — core API v2.36 |
| SDK header (build) | QAIRT 2.43.0.260128 (core API 2.32, system API 1.7) |
| Toolchain (host) | Bootlin aarch64 **gcc 10.3.0, glibc 2.34** (≤ 2.35) |
| Camera | CSI qua Qualcomm camera stack, lấy frame bằng GStreamer `qtiqmmfsrc` |

> Thiết bị **không có trình biên dịch** → phải **cross-compile trên host** rồi copy binary sang.

---

## 3. Cấu trúc repo

```
.
├── README.md                  # File này (tổng quan toàn dự án)
├── .gitignore
│
├── src/                       # Mã nguồn C
│   ├── face_det_lite.c        #   App 1: inference 1 ảnh
│   ├── cam_detect_web.c       #   App 2: camera + face + web MJPEG
│   ├── cam_yolo_web.c         #   App 3: camera + YOLOv8 + web MJPEG
│   └── yolo_inspect.c         #   Tool in I/O (shape/dtype/quant) của DLC/.bin
│
├── third_party/               # stb_image.h, stb_image_write.h (không cần OpenCV)
│
├── scripts/                   # Build & chạy
│   ├── build.sh               #   Cross-compile TẤT CẢ app -> build/
│   ├── run_on_device.sh       #   Chạy App 1 trên device
│   ├── run_cam_web.sh         #   Chạy App 2 trên device
│   ├── run_yolo_web.sh        #   Chạy App 3 trên device
│   └── verify_npu.sh          #   Chứng minh inference chạy trên NPU (HTP vs CPU)
│
├── model_compile/             # Tạo model QNN (w8a8) qua Qualcomm AI Hub
│   ├── compile_yolo_aihub.py  #   ONNX -> quantize w8a8 -> compile context binary
│   ├── calib_capture.sh       #   Chụp ảnh calibration từ camera (chạy trên device)
│   ├── requirements.txt
│   └── README.md
│
├── models/
│   └── yolov8_w8a8.bin        # Model YOLOv8 (QNN context binary, w8a8) cho QCS6490
│
├── assets/
│   └── result.png             # Ảnh demo App 1
│
└── docs/                      # Tài liệu chi tiết từng app
    ├── README_face_image.md
    ├── README_camera_web.md
    └── README_yolo_web.md
```

> Binary biên dịch nằm ở `build/` và **không** được track (xem `.gitignore`) — build lại từ source.
> File model `.dlc` của FaceDetLite nằm ở `../Model/` (do người dùng cung cấp từ AI Hub).

---

## 4. Toolchain & Build

### Lấy cross-compiler (một lần)
```bash
cd /tmp
curl -LO https://toolchains.bootlin.com/downloads/releases/toolchains/aarch64/tarballs/aarch64--glibc--stable-2021.11-1.tar.bz2
tar xf aarch64--glibc--stable-2021.11-1.tar.bz2
export TC=/tmp/aarch64--glibc--stable-2021.11-1
export SDK=/đường/dẫn/tới/qairt/2.43.0.260128
```

### Build cả 3 app
```bash
TC=$TC SDK=$SDK bash scripts/build.sh      # output -> build/
```
Tương đương (thủ công):
```bash
CC=$TC/bin/aarch64-linux-gcc ; INC="-I$SDK/include/QNN -Ithird_party"
$CC -O2 -std=gnu11 $INC src/face_det_lite.c  -o build/face_det_lite  -ldl -lm
$CC -O2 -std=gnu11 $INC src/cam_detect_web.c -o build/cam_detect_web -ldl -lm -lpthread
$CC -O2 -std=gnu11 $INC src/cam_yolo_web.c   -o build/cam_yolo_web   -ldl -lm -lpthread
$CC -O2 -std=gnu11 $INC src/yolo_inspect.c   -o build/yolo_inspect   -ldl
```

### Deploy
```bash
DEV=root@192.168.5.72            # mật khẩu thiết bị: <device-password>
ssh $DEV 'mkdir -p /opt/face_det_app'
scp build/* scripts/run_*.sh scripts/verify_npu.sh models/yolov8_w8a8.bin $DEV:/opt/face_det_app/
scp ../Model/face_det_lite-qnn_dlc-w8a8/face_det_lite.dlc $DEV:/opt/face_det_app/
```

**Biến môi trường runtime bắt buộc** (đã set trong các `run_*.sh`):
```sh
export LD_LIBRARY_PATH=/opt/qcom/qairt-latest/lib            # tìm libQnn*.so
export ADSP_LIBRARY_PATH=/opt/qcom/qairt-latest/lib/hexagon-v68/unsigned  # skel V68 cho DSP
```

---

## 5. App 1 — `face_det_lite` (ảnh tĩnh)
Phát hiện khuôn mặt trên 1 ảnh, ghi ảnh có vẽ box.
```bash
ssh root@192.168.5.72 'sh /opt/face_det_app/run_on_device.sh'
# -> /opt/face_det_app/result.png ; in toạ độ box + score
```
Chi tiết: [README_face_image.md](docs/README_face_image.md). Pipeline: resize 640×480 → grayscale luma →
HTP → sigmoid+maxpool peak → decode box (stride 8) → NMS.

## 6. App 2 — `cam_detect_web`
Camera realtime → phát hiện khuôn mặt → xem qua trình duyệt.
```bash
ssh root@192.168.5.72 'cd /opt/face_det_app && setsid sh run_cam_web.sh </dev/null >/tmp/cw.log 2>&1 &'
# Mở: http://192.168.5.72:8080/
```
Chi tiết: [README_camera_web.md](docs/README_camera_web.md).

## 7. App 3 — `cam_yolo_web`
Camera realtime → **YOLOv8** object detection (80 lớp COCO) → xem qua trình duyệt.
```bash
ssh root@192.168.5.72 'cd /opt/face_det_app && setsid sh run_yolo_web.sh </dev/null >/tmp/yolo.log 2>&1 &'
# Mở: http://192.168.5.72:8080/
# Dừng: pkill -9 -f cam_yolo_web; pkill -9 -f gst-launch
```
Endpoints: `/` (HTML), `/stream` (MJPEG), `/snapshot` (JPEG), `/dets` (JSON).
Chi tiết: [README_yolo_web.md](docs/README_yolo_web.md).

**I/O model YOLOv8** (model đã nhúng decode+sigmoid+argmax):
`image[1,3,640,640]` uint8 NCHW → `boxes[1,8400,4]` (xyxy), `scores[1,8400]`, `class_idx[1,8400]`.
Postprocess C: threshold (0.40) → NMS theo lớp (0.45) → nhãn COCO.

---

## 8. Model lấy từ đâu

- **FaceDetLite DLC**: tải sẵn từ Qualcomm AI Hub (`face_det_lite-qnn_dlc-w8a8`) — nạp trực tiếp bằng `systemDlcCreateFromFile`.
- **YOLOv8** (`models/yolov8_w8a8.bin`): các DLC/.bin YOLO cũ trên device **không tương thích** QNN runtime,
  nên tạo mới qua **Qualcomm AI Hub** bằng `qai-hub` (cần free API token):
  1. ONNX YOLOv8 (đã nhúng hậu xử lý) → upload.
  2. **Quantize job** w8a8 (INT8) với ảnh calibration chụp từ camera.
  3. **Compile job** `--target_runtime qnn_context_binary --quantize_io` → `yolov8_w8a8.bin`.
     (HTP **bắt buộc** I/O quantized; bản float bị từ chối / crash khi compose.)

  → Quy trình tự động hoá trong **[`model_compile/`](model_compile/README.md)**:
  ```bash
  pip install -r model_compile/requirements.txt
  qai-hub configure --api_token <FREE_TOKEN>
  # chụp calib trên device rồi:
  python3 model_compile/compile_yolo_aihub.py --onnx yolo.onnx --calib-dir ./calib \
      --device "Dragonwing RB3 Gen 2 Vision Kit" --out models/yolov8_w8a8.bin
  ```

Tool kiểm tra I/O bất kỳ DLC/.bin: `./yolo_inspect <model.dlc|model.bin>`.

---

## 9. Xác nhận chạy trên NPU
```bash
ssh root@192.168.5.72 'sh /opt/face_det_app/verify_npu.sh'
```
Bằng chứng: CDSP remoteproc `running`, nạp `libQnnHtpV68Stub.so` + `libcdsprpc.so` (FastRPC),
log có `VTCM Allocation` (bộ nhớ riêng Hexagon), và **HTP nhanh ~16× CPU** (vd face: 2.3 ms vs 37 ms).
App cũng in `Backend build id` qua `backendGetBuildId` và cho chọn backend bằng `QNN_BACKEND=libQnnCpu.so` để đối chiếu.

---

## 10. Khắc phục sự cố

| Vấn đề | Xử lý |
|--------|-------|
| `dlopen libQnnHtp.so ... not found` | set `LD_LIBRARY_PATH=/opt/qcom/qairt-latest/lib` |
| Lỗi finalize/execute trên DSP | thiếu `ADSP_LIBRARY_PATH` → `.../hexagon-v68/unsigned` |
| App tắt khi đóng SSH | chạy `setsid ./app … </dev/null &` (không dùng `nohup` đơn thuần) |
| `version GLIBC_2.xx not found` | toolchain build có glibc > 2.35 → dùng glibc ≤ 2.35 |
| DLC mở lỗi (`dlc open fail`) | DLC định dạng SNPE cũ — cần QNN DLC mới / context binary |
| `floating-point ... not supported` khi compile .bin | thêm `--quantize_io` + quantize w8a8 |
| Camera fps=0 | xem `/tmp/gst_cam.log`; `pkill gst-launch` rồi chạy lại |
| Cảnh báo `GBM_ERR ... PRIME_FD` | vô hại, frame vẫn chảy ~30fps |

---

## 11. Bảo mật
- **KHÔNG** commit secret. Mật khẩu thiết bị trong README đã thay bằng `<device-password>`.
- Remote git từng nhúng **GitHub PAT** và token **qai-hub** đã dùng khi build — **nên rotate** cả hai.
- Model `.bin/.dlc` không chứa secret.

---

## License / Credits
- QNN/QAIRT SDK © Qualcomm Technologies, Inc.
- `stb_image.h`, `stb_image_write.h` — public domain (Sean Barrett).
- Model FaceDetLite, YOLOv8-Detection — Qualcomm AI Hub.
