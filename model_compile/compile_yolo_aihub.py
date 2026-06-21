#!/usr/bin/env python3
"""
compile_yolo_aihub.py — Tao QNN context binary (w8a8) cho QCS6490 tu mot ONNX,
dung Qualcomm AI Hub (quantize + compile).

Pipeline:
  ONNX (da nhung hau xu ly) -> upload -> quantize w8a8 (INT8) voi calib
  -> compile --target_runtime qnn_context_binary --quantize_io -> tai ve .bin

Yeu cau:
  pip install -r requirements.txt
  qai-hub configure --api_token <FREE_TOKEN>     # hoac dat env QAI_HUB_API_TOKEN
  (token MIEN PHI lay tai https://aihub.qualcomm.com -> Settings -> API token)

Vi du:
  python3 compile_yolo_aihub.py \
      --onnx yolo.onnx --calib-dir ./calib --width 640 --height 640 \
      --device "Dragonwing RB3 Gen 2 Vision Kit" --out ../models/yolov8_w8a8.bin

Calib: thu muc chua cac file .rgb (raw RGB, width*height*3 bytes/uint8),
       tao bang scripts/calib_capture tren device (xem calib_capture.sh).
"""
import argparse, glob, os, sys
import numpy as np


def load_calib(calib_dir, w, h, input_name):
    files = sorted(glob.glob(os.path.join(calib_dir, "*.rgb")))
    if not files:
        sys.exit(f"[ERR] khong tim thay *.rgb trong {calib_dir}")
    samples = []
    for f in files:
        raw = np.fromfile(f, dtype=np.uint8)
        if raw.size != w * h * 3:
            print(f"  bo qua {f} (size {raw.size} != {w*h*3})")
            continue
        chw = np.transpose(raw.reshape(h, w, 3), (2, 0, 1)).astype(np.float32) / 255.0
        samples.append(chw[None, ...])  # [1,3,H,W], RGB [0,1]
    if not samples:
        sys.exit("[ERR] khong co mau calib hop le")
    print(f"[INFO] calib: {len(samples)} mau {samples[0].shape}")
    return {input_name: samples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="ONNX dau vao (co the dung external data)")
    ap.add_argument("--calib-dir", required=True, help="thu muc chua *.rgb")
    ap.add_argument("--device", default="Dragonwing RB3 Gen 2 Vision Kit")
    ap.add_argument("--input-name", default="image")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--out", default="../models/yolov8_w8a8.bin")
    ap.add_argument("--runtime", default="qnn_context_binary",
                    help="qnn_context_binary | qnn_dlc")
    args = ap.parse_args()

    import onnx
    import qai_hub as hub

    # 1) gop external data -> 1 file de upload
    single = args.onnx
    m = onnx.load(args.onnx, load_external_data=True)
    single = os.path.splitext(args.onnx)[0] + "_single.onnx"
    onnx.save_model(m, single, save_as_external_data=False)
    print(f"[INFO] consolidated ONNX -> {single}")

    dev = hub.Device(args.device)
    calib = load_calib(args.calib_dir, args.width, args.height, args.input_name)

    model = hub.upload_model(single)
    print(f"[INFO] uploaded model: {model.model_id}")

    # 2) quantize w8a8
    qjob = hub.submit_quantize_job(
        model, calibration_data=calib,
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT8,
        name="yolo_w8a8",
    )
    print(f"[INFO] quantize job: {qjob.url}")
    qjob.wait()
    if not qjob.get_status().success:
        sys.exit(f"[ERR] quantize FAILED: {qjob.get_status()}")
    qmodel = qjob.get_target_model()

    # 3) compile (quantize_io de I/O cung la uint8 -> HTP chap nhan)
    cjob = hub.submit_compile_job(
        qmodel, device=dev,
        name="yolo_w8a8_compile",
        options=f"--target_runtime {args.runtime} --quantize_io",
    )
    print(f"[INFO] compile job: {cjob.url}")
    cjob.wait()
    if not cjob.get_status().success:
        sys.exit(f"[ERR] compile FAILED: {cjob.get_status()}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cjob.get_target_model().download(args.out)
    print(f"[OK] DA TAI: {args.out}")
    print("    Copy len device va dung voi cam_yolo_web / yolo_inspect.")


if __name__ == "__main__":
    main()
