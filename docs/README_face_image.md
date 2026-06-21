# FaceDetLite — Phát hiện khuôn mặt bằng C trên Qualcomm RB3 Gen2 (QCS6490 / HTP V68)

Ứng dụng C thuần chạy model **FaceDetLite (Lightweight-Face-Detection)** định dạng **QNN DLC (w8a8)**
trên NPU **Hexagon V68 (HTP)** của Qualcomm Robotics **RB3 Gen2 — QCS6490 (qcm6490)**.

Đầu vào là một ảnh (`.jpg/.png/...`), đầu ra là ảnh có vẽ khung khuôn mặt (`result.png`) và danh sách
toạ độ box in ra màn hình.

```
[INFO] graphExecute OK (2.87 ms)          <-- thời gian inference trên NPU
[INFO] Phat hien 1 khuon mat (threshold=0.55):
   #1  box=[L=212 T=37 R=461 B=396] w=249 h=359 score=0.844
```

---

## 1. Kiến trúc & nguyên lý

```
        HOST (Ubuntu/WSL, x86_64)                         DEVICE (RB3 Gen2, aarch64)
 ┌──────────────────────────────────┐            ┌──────────────────────────────────────┐
 │ face_det_lite.c                   │  cross     │ /opt/face_det_app/face_det_lite (ELF)  │
 │ + stb_image / stb_image_write     │  compile   │                                        │
 │ + headers QNN (chỉ lúc build)     ├──────────► │  dlopen libQnnHtp.so   (backend HTP)   │
 │ aarch64 gcc (glibc 2.34)          │   scp      │  dlopen libQnnSystem.so(nạp DLC)        │
 └──────────────────────────────────┘            │         │                              │
                                                  │         ▼  Hexagon V68 NPU (skel .so)   │
                                                  │  face_det_lite.dlc  ───►  inference     │
                                                  └──────────────────────────────────────┘
```

**Điểm mấu chốt:** toàn bộ thư viện QNN được nạp động bằng `dlopen` lúc **chạy**. Vì vậy lúc **build**
chỉ cần *header* QNN + libc (`-ldl -lm`), **không** cần link `libQnn*.so`. Nhờ đó binary rất gọn và
không phụ thuộc đường dẫn SDK trên host.

### Luồng gọi QNN API (đúng thứ tự như SDK SampleApp)
```
QnnInterface_getProviders        (libQnnHtp.so)      → chọn backend QNN v2.x
QnnSystemInterface_getProviders  (libQnnSystem.so)   → chọn system interface v1.x
backendCreate → deviceCreate → contextCreate
systemDlcCreateFromFile → systemDlcComposeGraphs     → nạp graph từ DLC vào context
graphRetrieve → graphFinalize                        → biên dịch graph cho HTP (1 lần, ~0.7s)
graphExecute                                         → chạy inference (~3 ms)
contextFree / deviceFree / backendFree               → dọn dẹp
```

---

## 2. Thông tin model (từ `metadata.json`)

| Tensor    | Shape (NHWC)     | Kiểu  | scale      | zero_point | Ý nghĩa |
|-----------|------------------|-------|------------|------------|---------|
| `input`   | `[1,480,640,1]`  | uint8 | 0.00392157 | 0          | Ảnh xám, giá trị = mức xám 0..255 |
| `heatmap` | `[1,60,80,1]`    | uint8 | 0.0263916  | 191        | Bản đồ điểm tin cậy (logit) |
| `bbox`    | `[1,60,80,4]`    | uint8 | 0.2910421  | 10         | Hồi quy khoảng cách (l,t,r,b) theo ô lưới |
| `landmark`| `[1,60,80,10]`   | uint8 | 0.1509857  | 102        | 5 điểm mốc (x5,y5) |

- Lưới đầu ra **80×60**, `stride = 8` ⇒ `80×8 = 640`, `60×8 = 480` (đúng kích thước ảnh vào).
- Công thức dequant của QNN: `real = scale × (q + offset)` với `offset = -zero_point`
  (chương trình đọc trực tiếp `scale/offset` từ DLC lúc chạy, không hard-code).

---

## 3. Cấu trúc thư mục

```
face_det_app/
├── face_det_lite.c          # Mã nguồn C chính (toàn bộ logic)
├── build.sh                 # Script cross-compile trên host
├── run_on_device.sh         # Script chạy trên device (đặt biến môi trường + chạy)
├── third_party/
│   ├── stb_image.h          # Đọc ảnh JPEG/PNG (public domain, header-only)
│   └── stb_image_write.h    # Ghi ảnh PNG
├── face_det_lite            # Binary aarch64 sau khi build
├── result.png               # Ảnh kết quả (kéo về từ device)
└── README.md                # File này
```

File model & ảnh test (ngoài thư mục app):
```
../Model/face_det_lite-qnn_dlc-w8a8/face_det_lite.dlc
../Model/face_det_lite-qnn_dlc-w8a8/metadata.json
../Model/images.jpg
```

---

## 4. Yêu cầu môi trường

### 4.1. Trên HOST (máy build)
- Linux x86_64 (đã test trên WSL Ubuntu).
- **Cross-compiler aarch64 với glibc ≤ 2.35** (device dùng glibc 2.35).
  Bản đã dùng: **Bootlin `aarch64--glibc--stable-2021.11-1`** (gcc 10.3.0, glibc 2.34).
  Tải về:
  ```bash
  cd /tmp
  curl -LO https://toolchains.bootlin.com/downloads/releases/toolchains/aarch64/tarballs/aarch64--glibc--stable-2021.11-1.tar.bz2
  tar xf aarch64--glibc--stable-2021.11-1.tar.bz2
  ```
- Bộ **QAIRT/QNN SDK** để lấy header (đã có sẵn): `qairt/2.43.0.260128/include/QNN`.

> ⚠️ Vì sao không build trên device? Thiết bị RB3 Gen2 **không có** trình biên dịch C nào
> (`gcc/cc1/as/ld` đều thiếu, `opkg` không có feed). Do đó phải **cross-compile trên host**.
> Binary tạo ra yêu cầu glibc ≥ 2.34 nên chạy tốt trên device (glibc 2.35).

### 4.2. Trên DEVICE (RB3 Gen2)
- Đã cài sẵn QAIRT runtime tại `/opt/qcom/qairt-latest/`:
  - `lib/libQnnHtp.so`, `lib/libQnnSystem.so`, `lib/libQnnModelDlc.so`
  - `lib/hexagon-v68/unsigned/libQnnHtpV68Skel.so` (skel chạy trên DSP)
- OS: QCOM Reference Distro (Wayland), kernel 6.6.x, aarch64, glibc 2.35.
- Runtime QNN trên device: core API **v2.36** (tương thích header SDK v2.32 vì cùng major 2).

---

## 5. Build (trên HOST)

```bash
cd face_det_app

# Tùy chọn: chỉ đường dẫn toolchain / SDK (mặc định đã trỏ đúng trong build.sh)
export TC=/tmp/aarch64--glibc--stable-2021.11-1
export SDK=/đường/dẫn/tới/qairt/2.43.0.260128

bash build.sh
```

Lệnh build thực chất:
```bash
$TC/bin/aarch64-linux-gcc -O2 -std=gnu11 -Wall \
    -I"$SDK/include/QNN" -Ithird_party \
    face_det_lite.c -o face_det_lite \
    -ldl -lm -lpthread
```

Kiểm tra:
```bash
file face_det_lite
# ELF 64-bit LSB pie executable, ARM aarch64, ... interpreter /lib/ld-linux-aarch64.so.1
```

---

## 6. Deploy (HOST → DEVICE)

```bash
DEV=root@192.168.5.72          # mật khẩu: <device-password>
ssh $DEV 'mkdir -p /opt/face_det_app'

scp face_det_lite            $DEV:/opt/face_det_app/
scp run_on_device.sh         $DEV:/opt/face_det_app/
scp ../Model/face_det_lite-qnn_dlc-w8a8/face_det_lite.dlc $DEV:/opt/face_det_app/
scp ../Model/images.jpg      $DEV:/opt/face_det_app/
```

---

## 7. Chạy (trên DEVICE)

```bash
ssh root@192.168.5.72 'sh /opt/face_det_app/run_on_device.sh'
# Hạ ngưỡng để bắt nhiều mặt hơn:
ssh root@192.168.5.72 'sh /opt/face_det_app/run_on_device.sh 0.5'
```

Nội dung `run_on_device.sh`:
```sh
#!/bin/sh
QNN=/opt/qcom/qairt-latest/lib
export LD_LIBRARY_PATH=$QNN:$LD_LIBRARY_PATH          # tìm libQnnHtp/System/ModelDlc.so
export ADSP_LIBRARY_PATH=$QNN/hexagon-v68/unsigned     # tìm skel chạy trên Hexagon V68
cd /opt/face_det_app
./face_det_lite ./face_det_lite.dlc ./images.jpg ./result.png "$@"
```

> **Bắt buộc** đặt 2 biến môi trường trên:
> - `LD_LIBRARY_PATH` → để loader tìm thấy các `libQnn*.so`.
> - `ADSP_LIBRARY_PATH` → để HTP nạp **skel V68** lên DSP. Thiếu biến này sẽ lỗi khi finalize/execute.

### Cú pháp chương trình
```
face_det_lite <model.dlc> <input_image> <output_image> [threshold]
```
| Tham số        | Mặc định | Mô tả |
|----------------|----------|-------|
| `model.dlc`    | —        | Đường dẫn DLC |
| `input_image`  | —        | Ảnh vào (JPG/PNG, kích thước bất kỳ — sẽ tự resize 640×480) |
| `output_image` | —        | Ảnh ra (PNG, có vẽ box) |
| `threshold`    | `0.55`   | Ngưỡng điểm tin cậy (sau sigmoid). Giảm → nhiều mặt hơn |

Lấy ảnh kết quả về host để xem:
```bash
scp root@192.168.5.72:/opt/face_det_app/result.png .
```

---

## 8. Pipeline xử lý (chi tiết trong `face_det_lite.c`)

### 8.1. Tiền xử lý
1. `stbi_load` đọc ảnh → RGB (kích thước gốc).
2. **Resize bilinear** về `640×480`.
3. Chuyển **xám** theo công thức luma của skimage `rgb2gray`:
   `gray = 0.2125·R + 0.7154·G + 0.0721·B` (làm tròn 0..255).
4. Vì input scale = 1/255, zp = 0 ⇒ giá trị uint8 đưa vào model **chính là** mức xám.
   Buffer xếp dạng NHWC `[1,480,640,1]` (chính là mảng xám hàng-trước).

### 8.2. Inference
- Sao chép template tensor input/output từ graph (giữ đúng id/dims/dtype/quant).
- Trỏ `clientBuf` của input tới buffer xám, output tới buffer cấp phát.
- Gọi `graphExecute`.

### 8.3. Hậu xử lý (giống `qai_hub_models/face_det_lite`)
1. Dequant heatmap rồi **sigmoid**: `s = 1/(1+exp(-(q+offset)·scale))`.
2. **Max-pool 3×3** tìm đỉnh cục bộ (ô có giá trị ≥ mọi lân cận).
3. Với mỗi đỉnh có `s ≥ threshold`, decode box từ `bbox`:
   ```
   left   = (cx - bbox0) · stride
   top    = (cy - bbox1) · stride
   right  = (cx + bbox2) · stride
   bottom = (cy + bbox3) · stride        (stride = 8)
   ```
4. **NMS (IoU > 0.5)** gộp các box trùng. *(Bắt buộc thêm vì heatmap lượng tử hoá tạo
   “cao nguyên” nhiều ô cùng giá trị ⇒ nhiều đỉnh cho cùng 1 mặt; bản float gốc không cần NMS.)*
5. Vẽ khung xanh lá (dày 2px) lên ảnh 640×480 và ghi PNG.

---

## 8b. Xác nhận inference chạy trên NPU (Hexagon V68 HTP)

Chạy script kiểm chứng tự động trên device:
```bash
ssh root@192.168.5.72 'sh /opt/face_det_app/verify_npu.sh'
```
Script in ra 5 nhóm bằng chứng. Dưới đây là kết quả thực đo trên thiết bị:

**(1) Compute DSP đang chạy + kênh FastRPC tồn tại**
```
/sys/class/remoteproc/remoteprocX  state=running     # lõi Hexagon DSP hoạt động
/dev/fastrpc-cdsp                                     # kênh CPU <-> Compute DSP
```

**(2) Tiến trình nạp thư viện offload xuống DSP** (xem `/proc/<pid>/maps`)
```
libQnnHtpV68Stub.so     # stub HTP V68 (phía CPU, proxy tới skel trên DSP)
libcdsprpc.so           # FastRPC tới Compute DSP (Hexagon)
```

**(3) Dấu hiệu chỉ có ở NPU trong log** (CPU không bao giờ in ra):
- `rpcmem_init ... from libxdsprpc` — bộ nhớ chia sẻ FastRPC xuống DSP
- `VTCM Allocation` — **VTCM (Vector Tightly-Coupled Memory)** là bộ nhớ riêng của Hexagon NPU
- `Graph Sequencing for Target`, `DDR bandwidth summary` — trình biên dịch graph của HTP
- `Backend build id: v2.47.0...` (lấy bằng `backendGetBuildId`)

**(4) So sánh A/B (bằng chứng quyết định)** — cùng model, cùng ảnh:

| Backend | Lệnh | `graphExecute` | VTCM / FastRPC |
|---------|------|----------------|----------------|
| **HTP (NPU)** | mặc định | **~2.3 ms** | **Có** |
| **CPU** | `QNN_BACKEND=libQnnCpu.so` | **~37 ms** | **Không** |

→ HTP nhanh ~16×, và chỉ bản HTP mới có VTCM + FastRPC ⇒ khẳng định đang dùng Hexagon NPU.
Hai backend cho **cùng kết quả phát hiện** ⇒ tính đúng đắn được bảo toàn.

Chạy ép CPU để tự đối chiếu:
```bash
QNN_BACKEND=libQnnCpu.so sh /opt/face_det_app/run_on_device.sh
```

> Biến môi trường `QNN_BACKEND` chọn thư viện backend (mặc định `libQnnHtp.so`).
> Các lựa chọn có trên device: `libQnnHtp.so` (NPU), `libQnnCpu.so` (CPU), `libQnnGpu.so` (GPU).

---

## 9. Tinh chỉnh

| Muốn gì | Sửa ở đâu |
|---------|-----------|
| Nhiều/ít mặt hơn | Tham số `threshold` (CLI) — mặc định 0.55 |
| Độ gộp box NMS  | `nms(dets, ndet, 0.5f)` trong `main` (đổi 0.5) |
| Đổi màu/độ dày khung | Hàm `draw_rect()` |
| Vẽ thêm 5 landmark | Dùng buffer `lm` + dequant `(q+lm_o)*lm_s`, vị trí `(cx+lmx)·8, (cy+lmy)·8` |
| Đổi kích thước input | `IN_W`, `IN_H` (phải khớp shape model) |

---

## 10. Khắc phục sự cố

| Triệu chứng | Nguyên nhân / cách xử lý |
|-------------|--------------------------|
| `dlopen libQnnHtp.so: ... No such file` | Chưa set `LD_LIBRARY_PATH` tới `/opt/qcom/qairt-latest/lib` |
| Lỗi khi `graphFinalize`/`graphExecute` trên DSP | Thiếu `ADSP_LIBRARY_PATH` trỏ tới `hexagon-v68/unsigned` |
| `Khong tim thay backend interface QNN v2.x` | Runtime QNN trên device khác major với header build (phải cùng major 2) |
| `version GLIBC_2.xx not found` khi chạy | Toolchain build có glibc > 2.35; dùng toolchain glibc ≤ 2.35 |
| Không phát hiện mặt | Hạ `threshold` (vd 0.4); kiểm tra ảnh có mặt rõ, đủ sáng |
| Nhiều box chồng nhau | Tăng cường NMS hoặc giảm IoU threshold |
| `systemDlcCreateFromFile` lỗi | Sai đường dẫn DLC hoặc DLC không tương thích version runtime |

---

## 11. Hiệu năng & lưu ý

- `graphExecute`: **~3 ms/ảnh** trên HTP V68.
- `graphFinalize`: **~0.7 s** (tối ưu graph cho HTP) — **chi phí một lần**.
- Để xử lý nhiều ảnh / video: giữ nguyên context và **chỉ lặp lại bước `graphExecute`**
  (đổi dữ liệu trong `clientBuf` của input rồi gọi lại). Hoặc dùng
  `qnn-context-binary-generator` để cache context binary, lần sau nạp bằng
  `contextCreateFromBinary` để **bỏ qua** thời gian finalize.

---

## 12. Thông số môi trường đã kiểm chứng

| Hạng mục | Giá trị |
|----------|---------|
| Thiết bị | Qualcomm RB3 Gen2, SoC **QCS6490** (`qcm6490`) |
| NPU | Hexagon **V68** HTP |
| OS device | QCOM Reference Distro (Wayland), kernel 6.6.28, aarch64, glibc 2.35 |
| QNN runtime (device) | `/opt/qcom/qairt-latest`, core API v2.36 |
| SDK header (build) | QAIRT 2.43.0.260128 (core API 2.32, system API 1.7) |
| Toolchain | Bootlin aarch64 gcc 10.3.0, glibc 2.34 |
| Model | FaceDetLite, QNN DLC, w8a8 |
