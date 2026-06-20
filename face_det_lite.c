/* ============================================================================
 *  face_det_lite.c
 *
 *  Inference FaceDetLite (Lightweight-Face-Detection) DLC tren Qualcomm
 *  Robotics RB3 Gen2 (QCS6490 / qcm6490, Hexagon V68 HTP) bang QNN C API.
 *
 *  Luong:
 *    1. dlopen libQnnHtp.so  -> QnnInterface_getProviders        (backend HTP)
 *    2. dlopen libQnnSystem.so -> QnnSystemInterface_getProviders (DLC loader)
 *    3. backendCreate -> deviceCreate -> contextCreate
 *    4. systemDlcCreateFromFile -> systemDlcComposeGraphs (nap DLC vao context)
 *    5. graphRetrieve -> graphFinalize
 *    6. doc anh -> resize 640x480 -> grayscale uint8 (NHWC [1,480,640,1])
 *    7. graphExecute
 *    8. hau xu ly: dequant -> sigmoid heatmap -> maxpool 3x3 tim dinh
 *       -> decode bbox (threshold 0.55) -> ve box -> luu anh
 *
 *  Model I/O (tu metadata.json):
 *    input   : [1,480,640,1] uint8 grayscale, scale 1/255, zp 0
 *    heatmap : [1,60,80,1]  uint8, scale 0.02639, zp 191
 *    bbox    : [1,60,80,4]  uint8, scale 0.29104, zp 10
 *    landmark: [1,60,80,10] uint8, scale 0.15099, zp 102
 * ========================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <dlfcn.h>
#include <time.h>

#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/* ---- model geometry ---- */
#define IN_W      640
#define IN_H      480
#define GRID_W    80
#define GRID_H    60
#define STRIDE    8
#define MAX_DETS  512

/* ---- tien ich log ---- */
#define LOGI(...) do { fprintf(stdout, "[INFO] " __VA_ARGS__); fprintf(stdout, "\n"); } while (0)
#define LOGE(...) do { fprintf(stderr, "[ERR ] " __VA_ARGS__); fprintf(stderr, "\n"); } while (0)

#define QNN_CHECK(expr, msg)                                              \
    do {                                                                  \
        Qnn_ErrorHandle_t _e = (expr);                                    \
        if (_e != QNN_SUCCESS) {                                          \
            LOGE("%s (code=0x%llx)", msg, (unsigned long long)_e);        \
            goto fail;                                                    \
        }                                                                 \
    } while (0)

/* ============================================================================
 *  Tensor accessor (xu ly ca V1 va V2)
 * ========================================================================== */
static const char* t_name(const Qnn_Tensor_t* t) {
    return (t->version == QNN_TENSOR_VERSION_2) ? t->v2.name : t->v1.name;
}
static uint32_t t_rank(const Qnn_Tensor_t* t) {
    return (t->version == QNN_TENSOR_VERSION_2) ? t->v2.rank : t->v1.rank;
}
static const uint32_t* t_dims(const Qnn_Tensor_t* t) {
    return (t->version == QNN_TENSOR_VERSION_2) ? t->v2.dimensions : t->v1.dimensions;
}
static const Qnn_QuantizeParams_t* t_quant(const Qnn_Tensor_t* t) {
    return (t->version == QNN_TENSOR_VERSION_2) ? &t->v2.quantizeParams : &t->v1.quantizeParams;
}
static void t_set_rawbuf(Qnn_Tensor_t* t, void* data, uint32_t size) {
    if (t->version == QNN_TENSOR_VERSION_2) {
        t->v2.memType = QNN_TENSORMEMTYPE_RAW;
        t->v2.clientBuf.data = data;
        t->v2.clientBuf.dataSize = size;
    } else {
        t->v1.memType = QNN_TENSORMEMTYPE_RAW;
        t->v1.clientBuf.data = data;
        t->v1.clientBuf.dataSize = size;
    }
}
static uint32_t t_numel(const Qnn_Tensor_t* t) {
    uint32_t n = 1, r = t_rank(t);
    const uint32_t* d = t_dims(t);
    for (uint32_t i = 0; i < r; ++i) n *= d[i];
    return n;
}
/* lay scale & offset (QNN: real = scale * (q + offset)) */
static void t_scale_offset(const Qnn_Tensor_t* t, float* scale, int32_t* offset) {
    const Qnn_QuantizeParams_t* q = t_quant(t);
    *scale = 1.0f; *offset = 0;
    if (q->quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
        *scale  = q->scaleOffsetEncoding.scale;
        *offset = q->scaleOffsetEncoding.offset;
    } else if (q->quantizationEncoding == QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET) {
        *scale  = q->bwScaleOffsetEncoding.scale;
        *offset = q->bwScaleOffsetEncoding.offset;
    }
}

/* ============================================================================
 *  Bilinear resize RGB -> 640x480
 * ========================================================================== */
static void resize_rgb(const unsigned char* src, int sw, int sh,
                       unsigned char* dst, int dw, int dh) {
    float fx = (float)sw / dw;
    float fy = (float)sh / dh;
    for (int y = 0; y < dh; ++y) {
        float sy = (y + 0.5f) * fy - 0.5f;
        int y0 = (int)floorf(sy); float wy = sy - y0;
        int y1 = y0 + 1;
        if (y0 < 0) y0 = 0;
        if (y0 > sh - 1) y0 = sh - 1;
        if (y1 < 0) y1 = 0;
        if (y1 > sh - 1) y1 = sh - 1;
        for (int x = 0; x < dw; ++x) {
            float sx = (x + 0.5f) * fx - 0.5f;
            int x0 = (int)floorf(sx); float wx = sx - x0;
            int x1 = x0 + 1;
            if (x0 < 0) x0 = 0;
            if (x0 > sw - 1) x0 = sw - 1;
            if (x1 < 0) x1 = 0;
            if (x1 > sw - 1) x1 = sw - 1;
            for (int c = 0; c < 3; ++c) {
                float p00 = src[(y0 * sw + x0) * 3 + c];
                float p01 = src[(y0 * sw + x1) * 3 + c];
                float p10 = src[(y1 * sw + x0) * 3 + c];
                float p11 = src[(y1 * sw + x1) * 3 + c];
                float top = p00 + (p01 - p00) * wx;
                float bot = p10 + (p11 - p10) * wx;
                float v = top + (bot - top) * wy;
                int iv = (int)(v + 0.5f);
                if (iv < 0) iv = 0;
                if (iv > 255) iv = 255;
                dst[(y * dw + x) * 3 + c] = (unsigned char)iv;
            }
        }
    }
}

/* ve hinh chu nhat (vien) mau green tren anh RGB */
static void draw_rect(unsigned char* img, int w, int h,
                      int x0, int y0, int x1, int y1, int thick) {
    if (x0 > x1) { int t = x0; x0 = x1; x1 = t; }
    if (y0 > y1) { int t = y0; y0 = y1; y1 = t; }
    for (int t = 0; t < thick; ++t) {
        for (int x = x0; x <= x1; ++x) {
            int yt = y0 + t, yb = y1 - t;
            if (x >= 0 && x < w && yt >= 0 && yt < h) {
                img[(yt * w + x) * 3 + 0] = 0; img[(yt * w + x) * 3 + 1] = 255; img[(yt * w + x) * 3 + 2] = 0;
            }
            if (x >= 0 && x < w && yb >= 0 && yb < h) {
                img[(yb * w + x) * 3 + 0] = 0; img[(yb * w + x) * 3 + 1] = 255; img[(yb * w + x) * 3 + 2] = 0;
            }
        }
        for (int y = y0; y <= y1; ++y) {
            int xl = x0 + t, xr = x1 - t;
            if (y >= 0 && y < h && xl >= 0 && xl < w) {
                img[(y * w + xl) * 3 + 0] = 0; img[(y * w + xl) * 3 + 1] = 255; img[(y * w + xl) * 3 + 2] = 0;
            }
            if (y >= 0 && y < h && xr >= 0 && xr < w) {
                img[(y * w + xr) * 3 + 0] = 0; img[(y * w + xr) * 3 + 1] = 255; img[(y * w + xr) * 3 + 2] = 0;
            }
        }
    }
}

typedef struct { float x, y, r, b, score; } Det;

/* IoU giua 2 box [x,y,r,b] */
static float box_iou(const Det* a, const Det* b) {
    float ix0 = a->x > b->x ? a->x : b->x;
    float iy0 = a->y > b->y ? a->y : b->y;
    float ix1 = a->r < b->r ? a->r : b->r;
    float iy1 = a->b < b->b ? a->b : b->b;
    float iw = ix1 - ix0, ih = iy1 - iy0;
    if (iw <= 0 || ih <= 0) return 0.0f;
    float inter = iw * ih;
    float aa = (a->r - a->x) * (a->b - a->y);
    float bb = (b->r - b->x) * (b->b - b->y);
    float uni = aa + bb - inter;
    return uni > 0 ? inter / uni : 0.0f;
}

/* NMS tham lam: sap xep theo score giam dan, loai box trung IoU > thr.
 * Tra ve so box giu lai (ghi de mang dets). */
static int nms(Det* dets, int n, float iou_thr) {
    /* sap xep giam dan theo score (selection sort, n nho) */
    for (int i = 0; i < n; ++i) {
        int best = i;
        for (int j = i + 1; j < n; ++j)
            if (dets[j].score > dets[best].score) best = j;
        if (best != i) { Det tmp = dets[i]; dets[i] = dets[best]; dets[best] = tmp; }
    }
    int sup[MAX_DETS]; for (int i = 0; i < n; ++i) sup[i] = 0;
    Det keep[MAX_DETS]; int k = 0;
    for (int i = 0; i < n; ++i) {
        if (sup[i]) continue;
        keep[k++] = dets[i];
        for (int j = i + 1; j < n; ++j)
            if (!sup[j] && box_iou(&dets[i], &dets[j]) > iou_thr) sup[j] = 1;
    }
    for (int i = 0; i < k; ++i) dets[i] = keep[i];
    return k;
}

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ============================================================================
 *  getProviders typedefs
 * ========================================================================== */
typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(
    const QnnInterface_t*** providerList, uint32_t* numProviders);
typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn_t)(
    const QnnSystemInterface_t*** providerList, uint32_t* numProviders);

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: %s <model.dlc> <input_image> <output_image> [threshold=0.55]\n", argv[0]);
        return 1;
    }
    const char* dlc_path   = argv[1];
    const char* in_image   = argv[2];
    const char* out_image  = argv[3];
    float threshold        = (argc >= 5) ? (float)atof(argv[4]) : 0.55f;

    int ret = 1;
    void* htpLib = NULL;
    void* sysLib = NULL;

    QnnInterface_t          qIface = QNN_INTERFACE_INIT;          /* backend interface */
    QnnSystemInterface_t    sIface = QNN_SYSTEM_INTERFACE_INIT;   /* system interface  */
    int haveIface = 0, haveSys = 0;

    Qnn_BackendHandle_t backend = NULL;
    Qnn_DeviceHandle_t  device  = NULL;
    Qnn_ContextHandle_t context = NULL;
    QnnSystemDlc_Handle_t dlc   = NULL;

    QnnSystemContext_GraphInfo_t* graphs = NULL;
    uint32_t numGraphs = 0;
    uint32_t numOut = 0;   /* so output tensor (dung khi don dep) */

    unsigned char* img_src = NULL;
    unsigned char* img_rsz = NULL;
    unsigned char* in_buf  = NULL;
    Qnn_Tensor_t* inputs   = NULL;
    Qnn_Tensor_t* outputs  = NULL;
    void** out_bufs        = NULL;

    /* ---------------------------------------------------------------- *
     * 1. nap thu vien backend HTP + system, lay interface providers
     * ---------------------------------------------------------------- */
    htpLib = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_GLOBAL);
    {
        const char* backendLib = getenv("QNN_BACKEND");
        if (!backendLib || !backendLib[0]) backendLib = "libQnnHtp.so";
        if (htpLib) { dlclose(htpLib); htpLib = NULL; }
        htpLib = dlopen(backendLib, RTLD_NOW | RTLD_GLOBAL);
        if (!htpLib) { LOGE("dlopen %s: %s", backendLib, dlerror()); goto fail; }
        LOGI("Backend library: %s", backendLib);
    }
    sysLib = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_GLOBAL);
    if (!sysLib) { LOGE("dlopen libQnnSystem.so: %s", dlerror()); goto fail; }

    QnnInterfaceGetProvidersFn_t getIface =
        (QnnInterfaceGetProvidersFn_t)dlsym(htpLib, "QnnInterface_getProviders");
    QnnSystemInterfaceGetProvidersFn_t getSys =
        (QnnSystemInterfaceGetProvidersFn_t)dlsym(sysLib, "QnnSystemInterface_getProviders");
    if (!getIface || !getSys) { LOGE("dlsym getProviders that bai"); goto fail; }

    {
        const QnnInterface_t** provs = NULL; uint32_t n = 0;
        QNN_CHECK(getIface(&provs, &n), "QnnInterface_getProviders");
        for (uint32_t i = 0; i < n; ++i) {
            if (provs[i]->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR) {
                qIface = *provs[i]; haveIface = 1; break;
            }
        }
        if (!haveIface) { LOGE("Khong tim thay backend interface QNN v%d.x", QNN_API_VERSION_MAJOR); goto fail; }
        LOGI("Backend QNN core API v%u.%u",
             qIface.apiVersion.coreApiVersion.major, qIface.apiVersion.coreApiVersion.minor);
    }
    {
        const QnnSystemInterface_t** provs = NULL; uint32_t n = 0;
        QNN_CHECK(getSys(&provs, &n), "QnnSystemInterface_getProviders");
        for (uint32_t i = 0; i < n; ++i) {
            if (provs[i]->systemApiVersion.major == QNN_SYSTEM_API_VERSION_MAJOR) {
                sIface = *provs[i]; haveSys = 1; break;
            }
        }
        if (!haveSys) { LOGE("Khong tim thay system interface QNN"); goto fail; }
    }

    /* ---------------------------------------------------------------- *
     * 2. backendCreate -> deviceCreate -> contextCreate
     * ---------------------------------------------------------------- */
    QNN_CHECK(qIface.QNN_INTERFACE_VER_NAME.backendCreate(NULL, NULL, &backend), "backendCreate");
    LOGI("backendCreate OK");

    /* In dinh danh backend de xac nhan dang chay tren HTP (NPU) hay CPU */
    if (qIface.QNN_INTERFACE_VER_NAME.backendGetBuildId) {
        const char* bid = NULL;
        if (qIface.QNN_INTERFACE_VER_NAME.backendGetBuildId(&bid) == QNN_SUCCESS && bid)
            LOGI("Backend build id: %s", bid);
    }

    if (qIface.QNN_INTERFACE_VER_NAME.deviceCreate) {
        Qnn_ErrorHandle_t de = qIface.QNN_INTERFACE_VER_NAME.deviceCreate(NULL, NULL, &device);
        if (de != QNN_SUCCESS && de != QNN_DEVICE_NO_ERROR) {
            LOGI("deviceCreate khong dung (code=0x%llx) -> bo qua", (unsigned long long)de);
            device = NULL;
        } else {
            LOGI("deviceCreate OK");
        }
    }

    QNN_CHECK(qIface.QNN_INTERFACE_VER_NAME.contextCreate(backend, device, NULL, &context),
              "contextCreate");
    LOGI("contextCreate OK");

    /* ---------------------------------------------------------------- *
     * 3. nap DLC va compose graph vao context
     * ---------------------------------------------------------------- */
    QNN_CHECK(sIface.QNN_SYSTEM_INTERFACE_VER_NAME.systemDlcCreateFromFile(NULL, dlc_path, &dlc),
              "systemDlcCreateFromFile");
    LOGI("Nap DLC: %s", dlc_path);

    QNN_CHECK(sIface.QNN_SYSTEM_INTERFACE_VER_NAME.systemDlcComposeGraphs(
                  dlc, NULL, 0, backend, context, qIface,
                  QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1, &graphs, &numGraphs),
              "systemDlcComposeGraphs");
    if (numGraphs < 1) { LOGE("DLC khong co graph"); goto fail; }
    LOGI("Compose %u graph tu DLC", numGraphs);

    /* dung graph dau tien */
    QnnSystemContext_GraphInfoV1_t* gi = &graphs[0].graphInfoV1;
    Qnn_GraphHandle_t graph = NULL;
    QNN_CHECK(qIface.QNN_INTERFACE_VER_NAME.graphRetrieve(context, gi->graphName, &graph),
              "graphRetrieve");
    LOGI("Graph: '%s' | inputs=%u outputs=%u", gi->graphName, gi->numGraphInputs, gi->numGraphOutputs);

    QNN_CHECK(qIface.QNN_INTERFACE_VER_NAME.graphFinalize(graph, NULL, NULL), "graphFinalize");
    LOGI("graphFinalize OK");

    /* ---------------------------------------------------------------- *
     * 4. doc anh -> resize 640x480 -> grayscale uint8
     * ---------------------------------------------------------------- */
    int sw, sh, sc;
    img_src = stbi_load(in_image, &sw, &sh, &sc, 3);
    if (!img_src) { LOGE("Khong doc duoc anh: %s", in_image); goto fail; }
    LOGI("Anh goc: %dx%d (%d ch)", sw, sh, sc);

    img_rsz = (unsigned char*)malloc((size_t)IN_W * IN_H * 3);
    in_buf  = (unsigned char*)malloc((size_t)IN_W * IN_H);  /* grayscale */
    if (!img_rsz || !in_buf) { LOGE("malloc that bai"); goto fail; }
    resize_rgb(img_src, sw, sh, img_rsz, IN_W, IN_H);

    /* luma = 0.2125 R + 0.7154 G + 0.0721 B  (giong skimage rgb2gray) */
    for (int i = 0; i < IN_W * IN_H; ++i) {
        float g = 0.2125f * img_rsz[i*3+0] + 0.7154f * img_rsz[i*3+1] + 0.0721f * img_rsz[i*3+2];
        int q = (int)(g + 0.5f);
        if (q < 0) q = 0;
        if (q > 255) q = 255;
        in_buf[i] = (unsigned char)q;
    }

    /* ---------------------------------------------------------------- *
     * 5. chuan bi tensor input/output (sao chep tu template cua graph)
     * ---------------------------------------------------------------- */
    inputs  = (Qnn_Tensor_t*)malloc(sizeof(Qnn_Tensor_t) * gi->numGraphInputs);
    outputs = (Qnn_Tensor_t*)malloc(sizeof(Qnn_Tensor_t) * gi->numGraphOutputs);
    out_bufs = (void**)calloc(gi->numGraphOutputs, sizeof(void*));
    if (!inputs || !outputs || !out_bufs) { LOGE("malloc tensor that bai"); goto fail; }
    numOut = gi->numGraphOutputs;

    for (uint32_t i = 0; i < gi->numGraphInputs; ++i) {
        inputs[i] = gi->graphInputs[i];                 /* shallow copy template */
        t_set_rawbuf(&inputs[i], in_buf, (uint32_t)(IN_W * IN_H)); /* uint8, 1 byte/phan tu */
    }
    for (uint32_t i = 0; i < gi->numGraphOutputs; ++i) {
        outputs[i] = gi->graphOutputs[i];
        uint32_t nbytes = t_numel(&outputs[i]);          /* uint8 -> 1 byte/phan tu */
        out_bufs[i] = malloc(nbytes);
        if (!out_bufs[i]) { LOGE("malloc output that bai"); goto fail; }
        t_set_rawbuf(&outputs[i], out_bufs[i], nbytes);
    }

    /* ---------------------------------------------------------------- *
     * 6. graphExecute
     * ---------------------------------------------------------------- */
    double t0 = now_ms();
    QNN_CHECK(qIface.QNN_INTERFACE_VER_NAME.graphExecute(
                  graph, inputs, gi->numGraphInputs, outputs, gi->numGraphOutputs, NULL, NULL),
              "graphExecute");
    double t1 = now_ms();
    LOGI("graphExecute OK (%.2f ms)", t1 - t0);

    /* ---------------------------------------------------------------- *
     * 7. xac dinh output theo so kenh (last dim): 1=heatmap 4=bbox 10=landmark
     * ---------------------------------------------------------------- */
    unsigned char *hm = NULL, *bb = NULL, *lm = NULL;
    float hm_s = 1, bb_s = 1, lm_s = 1; int hm_o = 0, bb_o = 0, lm_o = 0;
    for (uint32_t i = 0; i < gi->numGraphOutputs; ++i) {
        uint32_t r = t_rank(&outputs[i]);
        const uint32_t* d = t_dims(&outputs[i]);
        uint32_t ch = d[r - 1];
        float s; int32_t o; t_scale_offset(&outputs[i], &s, &o);
        if      (ch == 1)  { hm = (unsigned char*)out_bufs[i]; hm_s = s; hm_o = o; }
        else if (ch == 4)  { bb = (unsigned char*)out_bufs[i]; bb_s = s; bb_o = o; }
        else if (ch == 10) { lm = (unsigned char*)out_bufs[i]; lm_s = s; lm_o = o; }
        LOGI("  out[%u] '%s' ch=%u scale=%.6f offset=%d", i, t_name(&outputs[i]), ch, s, o);
    }
    (void)lm; (void)lm_s; (void)lm_o;
    if (!hm || !bb) { LOGE("Thieu output heatmap/bbox"); goto fail; }

    /* ---------------------------------------------------------------- *
     * 8. hau xu ly: sigmoid + maxpool 3x3 tim dinh + decode bbox
     * ---------------------------------------------------------------- */
    static float hmf[GRID_H * GRID_W];
    for (int i = 0; i < GRID_H * GRID_W; ++i) {
        float logit = ((float)hm[i] + hm_o) * hm_s;
        hmf[i] = 1.0f / (1.0f + expf(-logit));
    }

    Det dets[MAX_DETS]; int ndet = 0;
    for (int cy = 0; cy < GRID_H; ++cy) {
        for (int cx = 0; cx < GRID_W; ++cx) {
            float v = hmf[cy * GRID_W + cx];
            if (v < threshold) continue;
            /* la dinh cuc bo neu >= moi lan can trong cua so 3x3 */
            int is_peak = 1;
            for (int dy = -1; dy <= 1 && is_peak; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int ny = cy + dy, nx = cx + dx;
                    if (ny < 0 || ny >= GRID_H || nx < 0 || nx >= GRID_W) continue;
                    if (hmf[ny * GRID_W + nx] > v) { is_peak = 0; break; }
                }
            }
            if (!is_peak) continue;

            int idx = cy * GRID_W + cx;
            float x = ((float)bb[idx*4+0] + bb_o) * bb_s;
            float y = ((float)bb[idx*4+1] + bb_o) * bb_s;
            float r = ((float)bb[idx*4+2] + bb_o) * bb_s;
            float b = ((float)bb[idx*4+3] + bb_o) * bb_s;
            if (ndet < MAX_DETS) {
                dets[ndet].x = (cx - x) * STRIDE;
                dets[ndet].y = (cy - y) * STRIDE;
                dets[ndet].r = (cx + r) * STRIDE;
                dets[ndet].b = (cy + b) * STRIDE;
                dets[ndet].score = v;
                ndet++;
            }
        }
    }

    LOGI("Tim duoc %d dinh tho; ap dung NMS (IoU 0.5)...", ndet);
    ndet = nms(dets, ndet, 0.5f);

    LOGI("Phat hien %d khuon mat (threshold=%.2f):", ndet, threshold);
    for (int i = 0; i < ndet; ++i) {
        int L = (int)(dets[i].x + 0.5f), T = (int)(dets[i].y + 0.5f);
        int R = (int)(dets[i].r + 0.5f), B = (int)(dets[i].b + 0.5f);
        printf("   #%d  box=[L=%d T=%d R=%d B=%d] w=%d h=%d score=%.3f\n",
               i + 1, L, T, R, B, R - L, B - T, dets[i].score);
        draw_rect(img_rsz, IN_W, IN_H, L, T, R, B, 2);
    }

    if (!stbi_write_png(out_image, IN_W, IN_H, 3, img_rsz, IN_W * 3)) {
        LOGE("Khong ghi duoc anh: %s", out_image);
        goto fail;
    }
    LOGI("Da luu anh ket qua: %s", out_image);

    ret = 0;

fail:
    /* don dep */
    if (out_bufs) {
        for (uint32_t i = 0; i < numOut; ++i) free(out_bufs[i]);
        free(out_bufs);
    }
    free(inputs); free(outputs);
    free(in_buf); free(img_rsz);
    if (img_src) stbi_image_free(img_src);
    if (dlc && haveSys) sIface.QNN_SYSTEM_INTERFACE_VER_NAME.systemDlcFree(dlc);
    if (context && haveIface) qIface.QNN_INTERFACE_VER_NAME.contextFree(context, NULL);
    if (device && haveIface && qIface.QNN_INTERFACE_VER_NAME.deviceFree)
        qIface.QNN_INTERFACE_VER_NAME.deviceFree(device);
    if (backend && haveIface) qIface.QNN_INTERFACE_VER_NAME.backendFree(backend);
    if (sysLib) dlclose(sysLib);
    if (htpLib) dlclose(htpLib);
    return ret;
}
