/* yolo_inspect.c - nap mot DLC qua QNN HTP va in I/O tensors (ten/shape/dtype/quant) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>
#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"

typedef Qnn_ErrorHandle_t (*GetIfaceFn)(const QnnInterface_t***, uint32_t*);
typedef Qnn_ErrorHandle_t (*GetSysFn)(const QnnSystemInterface_t***, uint32_t*);

static const char* dtype_name(Qnn_DataType_t d){
    switch(d){
        case QNN_DATATYPE_UFIXED_POINT_8: return "ufixed8";
        case QNN_DATATYPE_UFIXED_POINT_16:return "ufixed16";
        case QNN_DATATYPE_SFIXED_POINT_8: return "sfixed8";
        case QNN_DATATYPE_FLOAT_32: return "float32";
        case QNN_DATATYPE_FLOAT_16: return "float16";
        case QNN_DATATYPE_UINT_8: return "uint8";
        case QNN_DATATYPE_INT_32: return "int32";
        default: return "other";
    }
}
static void dump(const Qnn_Tensor_t* t, const char* tag, int i){
    const char* name = (t->version==QNN_TENSOR_VERSION_2)? t->v2.name : t->v1.name;
    uint32_t rank = (t->version==QNN_TENSOR_VERSION_2)? t->v2.rank : t->v1.rank;
    const uint32_t* d = (t->version==QNN_TENSOR_VERSION_2)? t->v2.dimensions : t->v1.dimensions;
    Qnn_DataType_t dt = (t->version==QNN_TENSOR_VERSION_2)? t->v2.dataType : t->v1.dataType;
    const Qnn_QuantizeParams_t* q = (t->version==QNN_TENSOR_VERSION_2)? &t->v2.quantizeParams : &t->v1.quantizeParams;
    printf("  %s[%d] '%s' dtype=%s rank=%u dims=[", tag, i, name?name:"(null)", dtype_name(dt), rank);
    for(uint32_t k=0;k<rank;k++) printf("%u%s", d[k], k+1<rank?",":"");
    printf("]");
    if(q->quantizationEncoding==QNN_QUANTIZATION_ENCODING_SCALE_OFFSET)
        printf(" scale=%.8f offset=%d", q->scaleOffsetEncoding.scale, q->scaleOffsetEncoding.offset);
    printf("\n");
}

int main(int argc, char** argv){
    if(argc<2){ printf("usage: %s model.dlc|model.bin\n", argv[0]); return 1; }
    const char* path = argv[1];
    size_t plen = strlen(path);
    int is_bin = (plen>4 && strcmp(path+plen-4, ".bin")==0);
    void* htp=dlopen("libQnnHtp.so",RTLD_NOW|RTLD_GLOBAL);
    void* sys=dlopen("libQnnSystem.so",RTLD_NOW|RTLD_GLOBAL);
    if(!htp||!sys){ printf("dlopen fail: %s\n", dlerror()); return 1; }
    GetIfaceFn gi=(GetIfaceFn)dlsym(htp,"QnnInterface_getProviders");
    GetSysFn   gs=(GetSysFn)dlsym(sys,"QnnSystemInterface_getProviders");
    QnnInterface_t qi; QnnSystemInterface_t si; int okq=0,oks=0;
    const QnnInterface_t** ip=NULL; uint32_t n=0; gi(&ip,&n);
    for(uint32_t i=0;i<n;i++) if(ip[i]->apiVersion.coreApiVersion.major==QNN_API_VERSION_MAJOR){qi=*ip[i];okq=1;break;}
    const QnnSystemInterface_t** sp=NULL; uint32_t m=0; gs(&sp,&m);
    for(uint32_t i=0;i<m;i++) if(sp[i]->systemApiVersion.major==QNN_SYSTEM_API_VERSION_MAJOR){si=*sp[i];oks=1;break;}
    if(!okq||!oks){ printf("no provider\n"); return 1; }

    Qnn_BackendHandle_t be=NULL; Qnn_DeviceHandle_t dev=NULL; Qnn_ContextHandle_t ctx=NULL;
    if(qi.QNN_INTERFACE_VER_NAME.backendCreate(NULL,NULL,&be)!=QNN_SUCCESS){printf("backendCreate fail\n");return 1;}
    if(qi.QNN_INTERFACE_VER_NAME.deviceCreate) qi.QNN_INTERFACE_VER_NAME.deviceCreate(NULL,NULL,&dev);

    if(is_bin){
        /* ---- context binary path ---- */
        FILE* f=fopen(path,"rb"); if(!f){printf("open bin fail\n");return 1;}
        fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
        uint8_t* buf=malloc(sz); if(fread(buf,1,sz,f)!=(size_t)sz){printf("read fail\n");return 1;} fclose(f);
        printf("binary size=%ld\n", sz);
        QnnSystemContext_Handle_t sc=NULL;
        si.QNN_SYSTEM_INTERFACE_VER_NAME.systemContextCreate(&sc);
        const QnnSystemContext_BinaryInfo_t* bi=NULL; Qnn_ContextBinarySize_t bisz=0;
        if(si.QNN_SYSTEM_INTERFACE_VER_NAME.systemContextGetBinaryInfo(sc,buf,sz,&bi,&bisz)!=QNN_SUCCESS){
            printf("getBinaryInfo fail\n"); return 1;
        }
        uint32_t ng=0; QnnSystemContext_GraphInfo_t* graphs=NULL;
        if(bi->version==QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1){ ng=bi->contextBinaryInfoV1.numGraphs; graphs=bi->contextBinaryInfoV1.graphs; }
        else if(bi->version==QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2){ ng=bi->contextBinaryInfoV2.numGraphs; graphs=bi->contextBinaryInfoV2.graphs; }
        else if(bi->version==QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3){ ng=bi->contextBinaryInfoV3.numGraphs; graphs=bi->contextBinaryInfoV3.graphs; }
        printf("graphs=%u (binary info v%d)\n", ng, bi->version);
        for(uint32_t g=0; g<ng; g++){
            QnnSystemContext_GraphInfoV1_t* gv=&graphs[g].graphInfoV1;
            printf("graph[%u] '%s' inputs=%u outputs=%u\n", g, gv->graphName, gv->numGraphInputs, gv->numGraphOutputs);
            for(uint32_t i=0;i<gv->numGraphInputs;i++)  dump(&gv->graphInputs[i], "in", i);
            for(uint32_t i=0;i<gv->numGraphOutputs;i++) dump(&gv->graphOutputs[i],"out", i);
        }
        return 0;
    }

    if(qi.QNN_INTERFACE_VER_NAME.contextCreate(be,dev,NULL,&ctx)!=QNN_SUCCESS){printf("contextCreate fail\n");return 1;}
    QnnSystemDlc_Handle_t dlc=NULL;
    if(si.QNN_SYSTEM_INTERFACE_VER_NAME.systemDlcCreateFromFile(NULL, path, &dlc)!=QNN_SUCCESS){printf("dlc open fail\n");return 1;}
    QnnSystemContext_GraphInfo_t* graphs=NULL; uint32_t ng=0;
    if(si.QNN_SYSTEM_INTERFACE_VER_NAME.systemDlcComposeGraphs(dlc,NULL,0,be,ctx,qi,
            QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1,&graphs,&ng)!=QNN_SUCCESS){printf("compose fail\n");return 1;}
    printf("graphs=%u\n", ng);
    for(uint32_t g=0; g<ng; g++){
        QnnSystemContext_GraphInfoV1_t* gv=&graphs[g].graphInfoV1;
        printf("graph[%u] '%s' inputs=%u outputs=%u\n", g, gv->graphName, gv->numGraphInputs, gv->numGraphOutputs);
        for(uint32_t i=0;i<gv->numGraphInputs;i++)  dump(&gv->graphInputs[i], "in", i);
        for(uint32_t i=0;i<gv->numGraphOutputs;i++) dump(&gv->graphOutputs[i],"out", i);
    }
    return 0;
}
