#!/usr/bin/env bash
# Cross-compile face_det_lite cho aarch64 (QCS6490 / qcm6490, glibc 2.35)
set -e

TC=${TC:-/tmp/opencode/aarch64--glibc--stable-2021.11-1}
SDK=${SDK:-/mnt/g/QDK/v2.43.0.260128/qairt/2.43.0.260128}
CC=$TC/bin/aarch64-linux-gcc
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "CC  = $CC"
echo "SDK = $SDK"

"$CC" -O2 -std=gnu11 -Wall \
    -I"$SDK/include/QNN" \
    -I"$HERE/third_party" \
    "$HERE/face_det_lite.c" \
    -o "$HERE/face_det_lite" \
    -ldl -lm -lpthread

echo "Build OK -> $HERE/face_det_lite"
file "$HERE/face_det_lite"
