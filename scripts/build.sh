#!/usr/bin/env bash
# Cross-compile tat ca app C cho aarch64 (QCS6490 / qcm6490, glibc 2.35).
# Yeu cau:
#   TC  = thu muc toolchain Bootlin aarch64 (gcc>=glibc 2.34, <=2.35)
#   SDK = thu muc QAIRT SDK (chua include/QNN)
# Vi du:
#   TC=/tmp/aarch64--glibc--stable-2021.11-1 SDK=/path/qairt/2.43.0.260128 bash scripts/build.sh
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TC=${TC:-/tmp/aarch64--glibc--stable-2021.11-1}
SDK=${SDK:-/mnt/g/QDK/v2.43.0.260128/qairt/2.43.0.260128}
CC="$TC/bin/aarch64-linux-gcc"
OUT="$ROOT/build"
INC="-I$SDK/include/QNN -I$ROOT/third_party"
CFLAGS="-O2 -std=gnu11 -Wall"

mkdir -p "$OUT"
echo "CC  = $CC"
echo "SDK = $SDK"
echo "OUT = $OUT"

build() { # <out> <src> <extra-libs>
    echo "  CC $2"
    "$CC" $CFLAGS $INC "$ROOT/src/$2" -o "$OUT/$1" $3
}

build face_det_lite  face_det_lite.c  "-ldl -lm"
build cam_detect_web cam_detect_web.c "-ldl -lm -lpthread"
build cam_yolo_web   cam_yolo_web.c   "-ldl -lm -lpthread"
build yolo_inspect   yolo_inspect.c   "-ldl"

echo "Build xong -> $OUT/"
ls -la "$OUT"
