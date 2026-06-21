#!/bin/sh
# =====================================================================
# verify_npu.sh  -  Xac nhan inference thuc su chay tren Hexagon NPU (HTP)
# Chay TREN DEVICE:  sh /opt/face_det_app/verify_npu.sh
# =====================================================================
QNN=/opt/qcom/qairt-latest/lib
export LD_LIBRARY_PATH=$QNN:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH=$QNN/hexagon-v68/unsigned
APP=/opt/face_det_app
cd "$APP" || exit 1

echo "############################################################"
echo "# 1) Trang thai Compute DSP (CDSP / remoteproc)"
echo "############################################################"
for r in /sys/class/remoteproc/remoteproc*; do
    [ -e "$r/name" ] || continue
    echo "  $r -> name=$(cat "$r/name" 2>/dev/null)  state=$(cat "$r/state" 2>/dev/null)"
done
echo
echo "  FastRPC device nodes (kenh giao tiep CPU<->DSP):"
ls -l /dev/fastrpc* 2>/dev/null | sed 's/^/    /'

echo
echo "############################################################"
echo "# 2) CDSP frequency truoc khi chay (devfreq)"
echo "############################################################"
CDSP_DEVFREQ=$(ls -d /sys/class/devfreq/*cdsp* 2>/dev/null | head -1)
[ -z "$CDSP_DEVFREQ" ] && CDSP_DEVFREQ=$(ls -d /sys/class/devfreq/*q6* 2>/dev/null | head -1)
if [ -n "$CDSP_DEVFREQ" ]; then
    echo "  $CDSP_DEVFREQ"
    echo "    cur_freq=$(cat "$CDSP_DEVFREQ/cur_freq" 2>/dev/null)"
    echo "    available=$(cat "$CDSP_DEVFREQ/available_frequencies" 2>/dev/null)"
else
    echo "  (khong tim thay devfreq cdsp - bo qua)"
fi

echo
echo "############################################################"
echo "# 3) Chay app tren HTP + bat libxdsprpc/skel trong /proc maps"
echo "############################################################"
# chay nen, lay PID, kiem tra cac thu vien FastRPC duoc nap
./face_det_lite ./face_det_lite.dlc ./images.jpg ./result_htp.png > /tmp/htp_run.log 2>&1 &
APP_PID=$!
# cho graph finalize, kiem tra maps nhieu lan
FOUND=""
i=0
while [ $i -lt 80 ]; do
    if [ -r /proc/$APP_PID/maps ]; then
        M=$(grep -E "libcdsprpc|libxdsprpc|libfastrpc|adsprpc|libQnnHtpV68" /proc/$APP_PID/maps 2>/dev/null | awk '{print $6}' | sort -u)
        [ -n "$M" ] && FOUND="$M"
        # CDSP freq khi dang chay
        [ -n "$CDSP_DEVFREQ" ] && CUR=$(cat "$CDSP_DEVFREQ/cur_freq" 2>/dev/null)
    fi
    kill -0 $APP_PID 2>/dev/null || break
    i=$((i+1)); sleep 0.1
done
wait $APP_PID
echo "  Thu vien FastRPC/HTP-stub da nap vao tien trinh:"
if [ -n "$FOUND" ]; then echo "$FOUND" | sed 's/^/    [DSP] /'; else echo "    (khong bat duoc - xem log)"; fi
[ -n "$CUR" ] && echo "  CDSP cur_freq luc chay = $CUR"

echo
echo "############################################################"
echo "# 4) Dau hieu HTP/DSP trong log (VTCM / FastRPC / DDR bw)"
echo "############################################################"
grep -iE "Backend library|build id|VTCM|Sequencing for Target|DDR bandwidth|rpcmem|graphExecute|Phat hien" /tmp/htp_run.log | sed 's/^/  /'

echo
echo "############################################################"
echo "# 5) So sanh: chay lai bang backend CPU (libQnnCpu.so)"
echo "############################################################"
QNN_BACKEND=libQnnCpu.so ./face_det_lite ./face_det_lite.dlc ./images.jpg ./result_cpu.png > /tmp/cpu_run.log 2>&1
echo "  --- CPU backend ---"
grep -iE "Backend library|build id|VTCM|rpcmem|graphExecute|Phat hien" /tmp/cpu_run.log | sed 's/^/  /'
echo
echo "  => Neu ban CPU KHONG co dong 'VTCM' / 'rpcmem'(FastRPC) va build id khac,"
echo "     thi ban HTP o tren that su chay tren Hexagon NPU."
echo "############################################################"
echo "# XONG. Ket qua HTP: result_htp.png | CPU: result_cpu.png"
echo "############################################################"
