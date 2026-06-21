/* ============================================================================
 *  cam_detect_web.c
 *
 *  Camera -> NPU (QNN HTP, Hexagon V68) -> Web (MJPEG over HTTP)
 *  tren Qualcomm RB3 Gen2 (QCS6490 / qcm6490).  Viet bang C thuan.
 *
 *  Luong:
 *    - Camera:  spawn `gst-launch-1.0 qtiqmmfsrc` ghi RGB tho ra 1 FIFO,
 *               chuong trinh C doc frame tu FIFO (khong dung OpenCV).
 *    - Detect:  QNN HTP nap DLC (mac dinh face_det_lite) chay tren NPU.
 *    - Web:     HTTP server da luong phat MJPEG (multipart/x-mixed-replace)
 *               + endpoint /dets (JSON) liet ke ket qua.
 *
 *  Thiet ke module-hoa: doi sang YOLO chi can thay 2 ham:
 *      detector_preprocess()  va  detector_postprocess()
 *  (xem khoi "DETECTOR" ben duoi).
 *
 *  Build:  cross-compile aarch64, can -ldl -lm -lpthread
 * ========================================================================== */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <dlfcn.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <pthread.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/* ---------------- cau hinh ---------------- */
#define CAP_W   640          /* camera capture = input YOLO = hien thi */
#define CAP_H   640
#define MAX_DETS 256
#define JPEG_QUALITY 80
#define FIFO_PATH "/tmp/cam_fifo"

/* ---------------- log ---------------- */
#define LOGI(...) do { fprintf(stdout,"[INFO] " __VA_ARGS__); fprintf(stdout,"\n"); fflush(stdout);} while(0)
#define LOGE(...) do { fprintf(stderr,"[ERR ] " __VA_ARGS__); fprintf(stderr,"\n"); } while(0)

typedef struct { float x, y, r, b, score; int cls; } Det;

/* ============================================================================
 *  PHAN 1: QNN backend + nap DLC  (tai dung tu face_det_lite)
 * ========================================================================== */
typedef Qnn_ErrorHandle_t (*GetIfaceFn)(const QnnInterface_t***, uint32_t*);
typedef Qnn_ErrorHandle_t (*GetSysFn)(const QnnSystemInterface_t***, uint32_t*);

static QnnInterface_t        g_qi;
static QnnSystemInterface_t  g_si;
static Qnn_BackendHandle_t   g_be = NULL;
static Qnn_DeviceHandle_t    g_dev = NULL;
static Qnn_ContextHandle_t   g_ctx = NULL;
static QnnSystemDlc_Handle_t g_dlc = NULL;
static Qnn_GraphHandle_t     g_graph = NULL;
static QnnSystemContext_GraphInfoV1_t* g_gi = NULL;

static Qnn_Tensor_t* g_inputs  = NULL;
static Qnn_Tensor_t* g_outputs = NULL;
static void**        g_outbufs = NULL;
static unsigned char* g_inbuf  = NULL;   /* buffer input (grayscale) */

/* tensor accessors (V1/V2) */
static uint32_t t_rank(const Qnn_Tensor_t* t){ return t->version==QNN_TENSOR_VERSION_2? t->v2.rank : t->v1.rank; }
static const uint32_t* t_dims(const Qnn_Tensor_t* t){ return t->version==QNN_TENSOR_VERSION_2? t->v2.dimensions : t->v1.dimensions; }
static const char* t_name(const Qnn_Tensor_t* t){ return t->version==QNN_TENSOR_VERSION_2? t->v2.name : t->v1.name; }
static void t_set_rawbuf(Qnn_Tensor_t* t, void* d, uint32_t s){
    if(t->version==QNN_TENSOR_VERSION_2){ t->v2.memType=QNN_TENSORMEMTYPE_RAW; t->v2.clientBuf.data=d; t->v2.clientBuf.dataSize=s; }
    else { t->v1.memType=QNN_TENSORMEMTYPE_RAW; t->v1.clientBuf.data=d; t->v1.clientBuf.dataSize=s; }
}
static uint32_t t_numel(const Qnn_Tensor_t* t){ uint32_t n=1,r=t_rank(t); const uint32_t* d=t_dims(t); for(uint32_t i=0;i<r;i++) n*=d[i]; return n; }
static void t_scale_offset(const Qnn_Tensor_t* t, float* s, int32_t* o){
    const Qnn_QuantizeParams_t* q = t->version==QNN_TENSOR_VERSION_2? &t->v2.quantizeParams : &t->v1.quantizeParams;
    *s=1.0f; *o=0;
    if(q->quantizationEncoding==QNN_QUANTIZATION_ENCODING_SCALE_OFFSET){ *s=q->scaleOffsetEncoding.scale; *o=q->scaleOffsetEncoding.offset; }
    else if(q->quantizationEncoding==QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET){ *s=q->bwScaleOffsetEncoding.scale; *o=q->bwScaleOffsetEncoding.offset; }
}

static int qnn_init(const char* bin_path){
    void* htp=dlopen("libQnnHtp.so",RTLD_NOW|RTLD_GLOBAL);
    void* sys=dlopen("libQnnSystem.so",RTLD_NOW|RTLD_GLOBAL);
    if(!htp||!sys){ LOGE("dlopen QNN: %s", dlerror()); return -1; }
    GetIfaceFn gi=(GetIfaceFn)dlsym(htp,"QnnInterface_getProviders");
    GetSysFn   gs=(GetSysFn)dlsym(sys,"QnnSystemInterface_getProviders");
    if(!gi||!gs){ LOGE("dlsym getProviders"); return -1; }
    const QnnInterface_t** ip=NULL; uint32_t n=0; int ok=0;
    if(gi(&ip,&n)!=QNN_SUCCESS) return -1;
    for(uint32_t i=0;i<n;i++) if(ip[i]->apiVersion.coreApiVersion.major==QNN_API_VERSION_MAJOR){ g_qi=*ip[i]; ok=1; break; }
    if(!ok){ LOGE("no backend provider"); return -1; }
    const QnnSystemInterface_t** sp=NULL; uint32_t m=0; ok=0;
    if(gs(&sp,&m)!=QNN_SUCCESS) return -1;
    for(uint32_t i=0;i<m;i++) if(sp[i]->systemApiVersion.major==QNN_SYSTEM_API_VERSION_MAJOR){ g_si=*sp[i]; ok=1; break; }
    if(!ok){ LOGE("no system provider"); return -1; }

    if(g_qi.QNN_INTERFACE_VER_NAME.backendCreate(NULL,NULL,&g_be)!=QNN_SUCCESS){ LOGE("backendCreate"); return -1; }
    if(g_qi.QNN_INTERFACE_VER_NAME.deviceCreate) g_qi.QNN_INTERFACE_VER_NAME.deviceCreate(NULL,NULL,&g_dev);

    /* ---- nap context binary (.bin) ---- */
    FILE* f=fopen(bin_path,"rb"); if(!f){ LOGE("open bin %s", bin_path); return -1; }
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    uint8_t* buf=malloc(sz);
    if(fread(buf,1,sz,f)!=(size_t)sz){ LOGE("read bin"); return -1; } fclose(f);

    QnnSystemContext_Handle_t sc=NULL;
    g_si.QNN_SYSTEM_INTERFACE_VER_NAME.systemContextCreate(&sc);
    const QnnSystemContext_BinaryInfo_t* bi=NULL; Qnn_ContextBinarySize_t bisz=0;
    if(g_si.QNN_SYSTEM_INTERFACE_VER_NAME.systemContextGetBinaryInfo(sc,buf,sz,&bi,&bisz)!=QNN_SUCCESS){ LOGE("getBinaryInfo"); return -1; }
    uint32_t ng=0; QnnSystemContext_GraphInfo_t* graphs=NULL;
    if(bi->version==QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1){ ng=bi->contextBinaryInfoV1.numGraphs; graphs=bi->contextBinaryInfoV1.graphs; }
    else if(bi->version==QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2){ ng=bi->contextBinaryInfoV2.numGraphs; graphs=bi->contextBinaryInfoV2.graphs; }
    else { ng=bi->contextBinaryInfoV3.numGraphs; graphs=bi->contextBinaryInfoV3.graphs; }
    if(ng<1){ LOGE("no graph in binary"); return -1; }
    g_gi=&graphs[0].graphInfoV1;   /* 5 truong dau chung cho V1/V2/V3 */

    if(g_qi.QNN_INTERFACE_VER_NAME.contextCreateFromBinary(g_be,g_dev,NULL,buf,sz,&g_ctx,NULL)!=QNN_SUCCESS){ LOGE("contextCreateFromBinary"); return -1; }
    free(buf);  /* QNN da copy */
    if(g_qi.QNN_INTERFACE_VER_NAME.graphRetrieve(g_ctx,g_gi->graphName,&g_graph)!=QNN_SUCCESS){ LOGE("graphRetrieve"); return -1; }
    LOGI("Graph '%s' inputs=%u outputs=%u (context binary, da finalize san)", g_gi->graphName, g_gi->numGraphInputs, g_gi->numGraphOutputs);

    /* setup tensors */
    g_inputs=malloc(sizeof(Qnn_Tensor_t)*g_gi->numGraphInputs);
    g_outputs=malloc(sizeof(Qnn_Tensor_t)*g_gi->numGraphOutputs);
    g_outbufs=calloc(g_gi->numGraphOutputs,sizeof(void*));
    g_inbuf=malloc(t_numel(&g_gi->graphInputs[0]));   /* uint8 */
    for(uint32_t i=0;i<g_gi->numGraphInputs;i++){
        g_inputs[i]=g_gi->graphInputs[i];
        t_set_rawbuf(&g_inputs[i], g_inbuf, t_numel(&g_inputs[i]));
    }
    for(uint32_t i=0;i<g_gi->numGraphOutputs;i++){
        g_outputs[i]=g_gi->graphOutputs[i];
        uint32_t nb=t_numel(&g_outputs[i]);
        g_outbufs[i]=malloc(nb);
        t_set_rawbuf(&g_outputs[i], g_outbufs[i], nb);
    }
    return 0;
}

static int qnn_execute(void){
    return g_qi.QNN_INTERFACE_VER_NAME.graphExecute(
        g_graph, g_inputs, g_gi->numGraphInputs, g_outputs, g_gi->numGraphOutputs, NULL, NULL)==QNN_SUCCESS ? 0 : -1;
}

/* ============================================================================
 *  PHAN 2: DETECTOR (face_det_lite). Doi YOLO -> sua 2 ham nay.
 * ========================================================================== */
/* ============================================================================
 *  PHAN 2: DETECTOR (YOLOv8 det, w8a8). Model da nhung decode+sigmoid+argmax:
 *     out boxes[1,8400,4] (xyxy, 640-space), scores[1,8400], class_idx[1,8400]
 * ========================================================================== */
static const char* COCO[80] = {
"person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
"fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
"elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
"skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
"wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
"broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
"dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
"toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"};
static const char* class_name(int c){ return (c>=0&&c<80)? COCO[c] : "obj"; }

static Qnn_DataType_t t_dtype(const Qnn_Tensor_t* t){ return t->version==QNN_TENSOR_VERSION_2? t->v2.dataType : t->v1.dataType; }

static float box_iou(const Det* a,const Det* b){
    float x0=a->x>b->x?a->x:b->x, y0=a->y>b->y?a->y:b->y;
    float x1=a->r<b->r?a->r:b->r, y1=a->b<b->b?a->b:b->b;
    float iw=x1-x0, ih=y1-y0; if(iw<=0||ih<=0) return 0;
    float inter=iw*ih, aa=(a->r-a->x)*(a->b-a->y), bb=(b->r-b->x)*(b->b-b->y);
    float u=aa+bb-inter; return u>0?inter/u:0;
}
/* NMS theo lop (chi triet tieu box cung class) */
static int nms(Det* d,int n,float thr){
    for(int i=0;i<n;i++){ int bi=i; for(int j=i+1;j<n;j++) if(d[j].score>d[bi].score) bi=j;
        if(bi!=i){ Det t=d[i]; d[i]=d[bi]; d[bi]=t; } }
    int sup[MAX_DETS]; for(int i=0;i<n;i++) sup[i]=0;
    Det keep[MAX_DETS]; int k=0;
    for(int i=0;i<n;i++){ if(sup[i])continue; keep[k++]=d[i];
        for(int j=i+1;j<n;j++) if(!sup[j]&&d[j].cls==d[i].cls&&box_iou(&d[i],&d[j])>thr) sup[j]=1; }
    for(int i=0;i<k;i++) d[i]=keep[i];
    return k;
}

/* tien xu ly: RGB CAP_WxCAP_H (interleaved) -> uint8 NCHW [1,3,640,640] vao g_inbuf */
static void detector_preprocess(const unsigned char* rgb){
    float s; int32_t o; t_scale_offset(&g_inputs[0],&s,&o);   /* scale=1/255, offset=0 */
    const int HW = CAP_W*CAP_H;
    for(int y=0;y<CAP_H;y++) for(int x=0;x<CAP_W;x++){
        int p=(y*CAP_W+x);
        for(int c=0;c<3;c++){
            float real = rgb[p*3+c]/255.0f;          /* normalize [0,1] */
            int q = (int)(real/s + 0.5f) - o;         /* QNN: real=s*(q+o) */
            if(q<0)q=0; if(q>255)q=255;
            g_inbuf[c*HW + p] = (unsigned char)q;      /* planar NCHW */
        }
    }
}

/* hau xu ly YOLOv8 -> dien Det (toa do 640-space). Tra ve so luong sau NMS. */
static int detector_postprocess(Det* out,int maxd,float threshold){
    unsigned char *bx=NULL,*sc=NULL,*ci=NULL;
    float bx_s=1,sc_s=1; int bx_o=0,sc_o=0; int N=0;
    for(uint32_t i=0;i<g_gi->numGraphOutputs;i++){
        uint32_t r=t_rank(&g_outputs[i]); const uint32_t* d=t_dims(&g_outputs[i]);
        float s; int32_t o; t_scale_offset(&g_outputs[i],&s,&o);
        if(r==3 && d[r-1]==4){ bx=g_outbufs[i]; bx_s=s; bx_o=o; N=d[1]; }     /* boxes */
        else if(r==2 && t_dtype(&g_outputs[i])==QNN_DATATYPE_UINT_8){ ci=g_outbufs[i]; } /* class_idx */
        else if(r==2){ sc=g_outbufs[i]; sc_s=s; sc_o=o; if(!N)N=d[1]; }        /* scores */
    }
    if(!bx||!sc||!ci||N<=0) return 0;
    int nd=0;
    for(int i=0;i<N && nd<maxd;i++){
        float score=sc_s*((float)sc[i]+sc_o);
        if(score<threshold) continue;
        int cls=ci[i];
        out[nd].x = bx_s*((float)bx[i*4+0]+bx_o);
        out[nd].y = bx_s*((float)bx[i*4+1]+bx_o);
        out[nd].r = bx_s*((float)bx[i*4+2]+bx_o);
        out[nd].b = bx_s*((float)bx[i*4+3]+bx_o);
        out[nd].score=score; out[nd].cls=cls;
        nd++;
    }
    return nms(out,nd,0.45f);
}

/* ve khung len anh RGB */
static void draw_rect(unsigned char* img,int w,int h,int x0,int y0,int x1,int y1,int thick){
    if(x0>x1){int t=x0;x0=x1;x1=t;} if(y0>y1){int t=y0;y0=y1;y1=t;}
    for(int t=0;t<thick;t++){
        for(int x=x0;x<=x1;x++){ int a=y0+t,b=y1-t;
            if(x>=0&&x<w&&a>=0&&a<h){img[(a*w+x)*3]=0;img[(a*w+x)*3+1]=255;img[(a*w+x)*3+2]=0;}
            if(x>=0&&x<w&&b>=0&&b<h){img[(b*w+x)*3]=0;img[(b*w+x)*3+1]=255;img[(b*w+x)*3+2]=0;} }
        for(int y=y0;y<=y1;y++){ int a=x0+t,b=x1-t;
            if(y>=0&&y<h&&a>=0&&a<w){img[(y*w+a)*3]=0;img[(y*w+a)*3+1]=255;img[(y*w+a)*3+2]=0;}
            if(y>=0&&y<h&&b>=0&&b<w){img[(y*w+b)*3]=0;img[(y*w+b)*3+1]=255;img[(y*w+b)*3+2]=0;} }
    }
}

/* ============================================================================
 *  PHAN 3: frame chia se (annotated JPEG + danh sach det) cho HTTP
 * ========================================================================== */
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_cond = PTHREAD_COND_INITIALIZER;
static unsigned char* g_jpeg = NULL;   /* JPEG moi nhat */
static size_t g_jpeg_sz = 0;
static uint64_t g_frame_id = 0;
static Det g_dets[MAX_DETS]; static int g_ndet=0;
static volatile int g_run = 1;
static double g_fps = 0;

typedef struct { unsigned char* d; size_t n,cap; } membuf;
static void mem_w(void* ctx,void* p,int n){ membuf* m=ctx;
    if(m->n+n>m->cap){ m->cap=(m->n+n)*2+1024; m->d=realloc(m->d,m->cap);} memcpy(m->d+m->n,p,n); m->n+=n; }

static double now_ms(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec*1000.0+ts.tv_nsec/1e6; }

/* ============================================================================
 *  PHAN 4: camera qua FIFO + GStreamer
 * ========================================================================== */
static int start_camera(void){
    unlink(FIFO_PATH);
    if(mkfifo(FIFO_PATH,0666)!=0 && errno!=EEXIST){ LOGE("mkfifo: %s", strerror(errno)); return -1; }
    char cmd[1024];
    snprintf(cmd,sizeof(cmd),
        "gst-launch-1.0 -q qtiqmmfsrc camera=0 ! "
        "video/x-raw,width=1280,height=720,format=NV12 ! "
        "videoconvert ! videoscale ! video/x-raw,format=RGB,width=%d,height=%d ! "
        "filesink location=%s buffer-mode=2 > /tmp/gst_cam.log 2>&1 &",
        CAP_W, CAP_H, FIFO_PATH);
    LOGI("Khoi dong camera: %s", cmd);
    if(system(cmd)!=0){ LOGE("system(gst) loi"); }
    int fd=open(FIFO_PATH,O_RDONLY);   /* block toi khi gst mo dau ghi */
    if(fd<0){ LOGE("open fifo: %s", strerror(errno)); return -1; }
    return fd;
}
static int read_full(int fd, unsigned char* buf, size_t n){
    size_t got=0;
    while(got<n){ ssize_t r=read(fd,buf+got,n-got);
        if(r<0){ if(errno==EINTR) continue; return -1; }
        if(r==0) return -1;  /* EOF: gst dong */
        got+=r; }
    return 0;
}

static void* capture_thread(void* arg){
    (void)arg;
    int fd=start_camera();
    if(fd<0){ g_run=0; return NULL; }
    unsigned char* rgb=malloc(CAP_W*CAP_H*3);
    Det dets[MAX_DETS];
    double last=now_ms(); int fcount=0;
    while(g_run){
        if(read_full(fd,rgb,CAP_W*CAP_H*3)!=0){ LOGE("doc camera that bai/EOF"); break; }
        /* inference tren NPU */
        detector_preprocess(rgb);
        if(qnn_execute()!=0){ LOGE("graphExecute loi"); continue; }
        int nd=detector_postprocess(dets,MAX_DETS,0.40f);
        /* ve box */
        for(int i=0;i<nd;i++)
            draw_rect(rgb,CAP_W,CAP_H,(int)dets[i].x,(int)dets[i].y,(int)dets[i].r,(int)dets[i].b,2);
        /* encode JPEG vao bo nho */
        membuf mb={0}; stbi_write_jpg_to_func(mem_w,&mb,CAP_W,CAP_H,3,rgb,JPEG_QUALITY);
        /* dem fps */
        fcount++; double t=now_ms(); if(t-last>=1000.0){ g_fps=fcount*1000.0/(t-last); fcount=0; last=t; }
        /* cap nhat frame chia se */
        pthread_mutex_lock(&g_lock);
        free(g_jpeg); g_jpeg=mb.d; g_jpeg_sz=mb.n;
        memcpy(g_dets,dets,sizeof(Det)*nd); g_ndet=nd;
        g_frame_id++;
        pthread_cond_broadcast(&g_cond);
        pthread_mutex_unlock(&g_lock);
    }
    free(rgb); close(fd);
    g_run=0;
    return NULL;
}

/* ============================================================================
 *  PHAN 5: HTTP server (MJPEG + trang HTML + /dets JSON)
 * ========================================================================== */
static int send_all(int fd,const void* buf,size_t n){
    const char* p=buf; size_t s=0;
    while(s<n){ ssize_t w=send(fd,p+s,n-s,MSG_NOSIGNAL); if(w<=0) return -1; s+=w; } return 0;
}

static const char* HTML_PAGE =
"<!doctype html><html><head><meta charset=utf-8><title>RB3 NPU Detection</title>"
"<style>body{background:#111;color:#eee;font-family:sans-serif;margin:0;padding:12px}"
"h2{margin:6px 0}#wrap{display:flex;gap:16px;flex-wrap:wrap}"
"img{border:2px solid #2d2;border-radius:6px;max-width:100%}"
"#side{min-width:240px}.d{background:#1d1d1d;padding:6px 10px;margin:4px 0;border-left:3px solid #2d2;border-radius:4px}"
"#fps{color:#2d2;font-weight:bold}</style></head><body>"
"<h2>Qualcomm RB3 Gen2 — NPU (HTP V68) — YOLOv8 Object Detection</h2>"
"<div id=wrap><div><img src=\"/stream\" width=640 height=640></div>"
"<div id=side><div>FPS: <span id=fps>-</span></div><h3>Detections</h3><div id=dets></div></div></div>"
"<script>setInterval(async()=>{try{let r=await fetch('/dets');let j=await r.json();"
"document.getElementById('fps').textContent=j.fps.toFixed(1);"
"document.getElementById('dets').innerHTML=j.dets.map((d,i)=>"
"`<div class=d>#${i+1} <b>${d.cls}</b> ${(d.score*100).toFixed(1)}%<br>"
"[x:${d.x|0} y:${d.y|0} w:${(d.r-d.x)|0} h:${(d.b-d.y)|0}]</div>`).join('')||'<i>none</i>';"
"}catch(e){}},400);</script></body></html>";

static void handle_client(int cfd){
    char req[2048]; ssize_t n=recv(cfd,req,sizeof(req)-1,0);
    if(n<=0){ close(cfd); return; }
    req[n]=0;
    char method[8]={0}, path[256]={0};
    sscanf(req,"%7s %255s",method,path);

    if(strcmp(path,"/")==0){
        char hdr[256]; int len=strlen(HTML_PAGE);
        snprintf(hdr,sizeof(hdr),"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: %d\r\nConnection: close\r\n\r\n",len);
        send_all(cfd,hdr,strlen(hdr)); send_all(cfd,HTML_PAGE,len); close(cfd); return;
    }
    if(strcmp(path,"/dets")==0){
        char body[8192]; int p=0;
        pthread_mutex_lock(&g_lock);
        p+=snprintf(body+p,sizeof(body)-p,"{\"fps\":%.2f,\"dets\":[",g_fps);
        for(int i=0;i<g_ndet;i++) p+=snprintf(body+p,sizeof(body)-p,
            "%s{\"cls\":\"%s\",\"score\":%.3f,\"x\":%.1f,\"y\":%.1f,\"r\":%.1f,\"b\":%.1f}",
            i?",":"", class_name(g_dets[i].cls), g_dets[i].score, g_dets[i].x, g_dets[i].y, g_dets[i].r, g_dets[i].b);
        p+=snprintf(body+p,sizeof(body)-p,"]}");
        pthread_mutex_unlock(&g_lock);
        char hdr[256];
        snprintf(hdr,sizeof(hdr),"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n",p);
        send_all(cfd,hdr,strlen(hdr)); send_all(cfd,body,p); close(cfd); return;
    }
    if(strcmp(path,"/snapshot")==0){
        pthread_mutex_lock(&g_lock);
        size_t sz=g_jpeg_sz; unsigned char* cpy=NULL;
        if(sz){ cpy=malloc(sz); memcpy(cpy,g_jpeg,sz); }
        pthread_mutex_unlock(&g_lock);
        if(!cpy){ const char* e="HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n"; send_all(cfd,e,strlen(e)); close(cfd); return; }
        char hdr[200]; snprintf(hdr,sizeof(hdr),"HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",sz);
        send_all(cfd,hdr,strlen(hdr)); send_all(cfd,cpy,sz); free(cpy); close(cfd); return;
    }
    if(strcmp(path,"/stream")==0){
        const char* hdr="HTTP/1.1 200 OK\r\nConnection: close\r\n"
            "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
        if(send_all(cfd,hdr,strlen(hdr))!=0){ close(cfd); return; }
        uint64_t last=0;
        while(g_run){
            pthread_mutex_lock(&g_lock);
            while(g_run && g_frame_id==last) pthread_cond_wait(&g_cond,&g_lock);
            last=g_frame_id;
            size_t sz=g_jpeg_sz; unsigned char* cpy=NULL;
            if(sz){ cpy=malloc(sz); memcpy(cpy,g_jpeg,sz); }
            pthread_mutex_unlock(&g_lock);
            if(!cpy) continue;
            char part[128];
            int ph=snprintf(part,sizeof(part),"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %zu\r\n\r\n",sz);
            if(send_all(cfd,part,ph)!=0 || send_all(cfd,cpy,sz)!=0 || send_all(cfd,"\r\n",2)!=0){ free(cpy); break; }
            free(cpy);
        }
        close(cfd); return;
    }
    const char* nf="HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
    send_all(cfd,nf,strlen(nf)); close(cfd);
}
static void* client_thread(void* a){ int fd=(int)(intptr_t)a; handle_client(fd); return NULL; }

int main(int argc,char** argv){
    if(argc<2){ printf("Usage: %s <model.bin> [port=8080]\n",argv[0]); return 1; }
    int port = argc>=3? atoi(argv[2]) : 8080;
    signal(SIGPIPE,SIG_IGN);

    if(qnn_init(argv[1])!=0){ LOGE("Khong khoi tao duoc QNN/DLC"); return 1; }

    pthread_t cap; pthread_create(&cap,NULL,capture_thread,NULL);

    int sfd=socket(AF_INET,SOCK_STREAM,0);
    int opt=1; setsockopt(sfd,SOL_SOCKET,SO_REUSEADDR,&opt,sizeof(opt));
    struct sockaddr_in addr={0}; addr.sin_family=AF_INET; addr.sin_addr.s_addr=INADDR_ANY; addr.sin_port=htons(port);
    if(bind(sfd,(struct sockaddr*)&addr,sizeof(addr))!=0){ LOGE("bind %d: %s",port,strerror(errno)); return 1; }
    listen(sfd,8);
    LOGI("==> Mo trinh duyet: http://<IP-thiet-bi>:%d/", port);

    while(g_run){
        struct sockaddr_in ca; socklen_t cl=sizeof(ca);
        int cfd=accept(sfd,(struct sockaddr*)&ca,&cl);
        if(cfd<0){ if(errno==EINTR) continue; break; }
        pthread_t t; pthread_create(&t,NULL,client_thread,(void*)(intptr_t)cfd); pthread_detach(t);
    }
    g_run=0; pthread_join(cap,NULL);
    close(sfd);
    return 0;
}
