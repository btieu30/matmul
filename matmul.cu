/*
 * matmul.cu — Massively Parallel Matrix Multiplication
 * MPI + CUDA with:
 *   - Custom tiled shared-memory GEMM kernel
 *   - Four precision modes: FP64, FP32, TF32, FP16+TC
 *   - MPI I/O checkpointing to NVMe
 *   - POWER9 cycle-counter instrumentation
 *
 * Usage:
 *   mpirun ./matmul N proc_rows proc_cols kernel [prec=0] [ckpt]
 *
 *   kernel: 0 = custom tiled shared-memory GEMM
 *           1 = cuBLAS
 *
 *   prec: 0 = FP64
 *         1 = FP32
 *         2 = TF32
 *         3 = FP16+TC
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include "clockcycle.h"

#define KERN_CUSTOM 0
#define KERN_CUBLAS 1

#define PREC_FP64 0
#define PREC_FP32 1
#define PREC_TF32 2
#define PREC_FP16 3

#define TILE_DIM 32

#define PANEL_B 128
#define CKPT_FREQ 16
#define CLOCK_RATE 512000000ULL

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr,"CUDA error %s:%d  %s\n", \
                __FILE__,__LINE__,cudaGetErrorString(_e)); \
        MPI_Abort(MPI_COMM_WORLD,1); } } while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t _s = (call); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr,"cuBLAS error %s:%d  status=%d\n", \
                __FILE__,__LINE__,(int)_s); \
        MPI_Abort(MPI_COMM_WORLD,1); } } while(0)

typedef struct {
    uint64_t bcast_a, bcast_b, h2d, compute, d2h, io, total;
} Timers;

static double cyc2s(uint64_t c) { return (double)c / (double)CLOCK_RATE; }

__global__ void custom_gemm_fp64(const double * __restrict__ A, const double * __restrict__ B, double * __restrict__ C, int M, int K, int N) {
    __shared__ double sA[TILE_DIM][TILE_DIM];
    __shared__ double sB[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    double acc = 0.0;

    int nStrips = (K + TILE_DIM - 1) / TILE_DIM;

    for (int strip = 0; strip < nStrips; strip++) {
        int aCol = strip * TILE_DIM + threadIdx.x;
        int bRow = strip * TILE_DIM + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0;
        sB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++)
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] += acc;
}

__global__ void custom_gemm_fp32(const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C, int M, int K, int N) {
    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    float acc = 0.0f;

    int nStrips = (K + TILE_DIM - 1) / TILE_DIM;
    for (int strip = 0; strip < nStrips; strip++) {
        int aCol = strip * TILE_DIM + threadIdx.x;
        int bRow = strip * TILE_DIM + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++)
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] += acc;
}

__global__ void d2f_cast(const double*src, float *dst, int n){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) dst[i] = (float)src[i]; }

__global__ void d2h_cast(const double*src, __half *dst, int n){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) dst[i] = __double2half(src[i]); }

__global__ void f2d_cast(const float *src, double*dst, int n){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) dst[i] = (double)src[i]; }

static void ckpt_write(const char *fname, const double *C, int lr, int lc, int N, int mr, int mc, uint64_t *cyc) {
    MPI_File fh; 
    MPI_Status st;
    uint64_t t0 = clock_now();

    int err = MPI_File_open(MPI_COMM_WORLD, fname, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    
    if (err != MPI_SUCCESS) {
        if (mr == 0 && mc == 0) fprintf(stderr, "Failed to open file: %s\n", fname);
        return;
    }

    MPI_Offset off = ((MPI_Offset)mr * lr * N + (MPI_Offset)mc * lc) * sizeof(double);
    
    err = MPI_File_write_at(fh, off, C, lr * lc, MPI_DOUBLE, &st);
    
    if (err != MPI_SUCCESS) {
        char err_str[MPI_MAX_ERROR_STRING];
        int len;
        MPI_Error_string(err, err_str, &len);
        fprintf(stderr, "Rank %d,%d Write Error: %s\n", mr, mc, err_str);
    }

    MPI_File_close(&fh);
    *cyc += clock_now() - t0;
}

static void local_gemm(cublasHandle_t hnd, int kernel, int prec, int lr, int pw, int lc, double *dpA64, double *dpB64, double *dC64, float  *dpA32, float  *dpB32, float  *dC32, __half *dpAh,  __half *dpBh) {
    if (kernel == KERN_CUSTOM) {
        dim3 block(TILE_DIM, TILE_DIM);
        dim3 grid((lc + TILE_DIM-1)/TILE_DIM, (lr + TILE_DIM-1)/TILE_DIM);

        if (prec == PREC_FP64)
            custom_gemm_fp64<<<grid, block>>>(dpA64, dpB64, dC64, lr, pw, lc);
        else
            custom_gemm_fp32<<<grid, block>>>(dpA32, dpB32, dC32, lr, pw, lc);

        CUDA_CHECK(cudaGetLastError());

    } else {
        const double a64=1.0, b64=1.0;
        const float  a32=1.f, b32=1.f;

        switch (prec) {
        case PREC_FP64:
            CUBLAS_CHECK(cublasDgemm(hnd, CUBLAS_OP_N, CUBLAS_OP_N, lc, lr, pw, &a64, dpB64, lc, dpA64, pw, &b64, dC64, lc));
            break;
        case PREC_FP32:
            CUBLAS_CHECK(cublasSgemm(hnd, CUBLAS_OP_N, CUBLAS_OP_N, lc, lr, pw, &a32, dpB32, lc, dpA32, pw, &b32, dC32, lc));
            break;
        case PREC_TF32:
            CUBLAS_CHECK(cublasGemmEx(hnd,
                CUBLAS_OP_N, CUBLAS_OP_N, lc, lr, pw, &a32,
                dpB32, CUDA_R_32F, lc,
                dpA32, CUDA_R_32F, pw,
                &b32,  dC32, CUDA_R_32F, lc,
                CUBLAS_COMPUTE_32F_FAST_TF32,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            break;
        case PREC_FP16:
            CUBLAS_CHECK(cublasGemmEx(hnd,
                CUBLAS_OP_N, CUBLAS_OP_N, lc, lr, pw, &a32,
                dpBh, CUDA_R_16F, lc,
                dpAh, CUDA_R_16F, pw,
                &b32, dC32, CUDA_R_32F, lc,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            break;
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int ws, wr;
    MPI_Comm_size(MPI_COMM_WORLD, &ws);
    MPI_Comm_rank(MPI_COMM_WORLD, &wr);

    if (argc < 5) {
        if (!wr) fprintf(stderr,
            "Usage: %s N pr pc kernel [prec=0] [ckpt]\n"
            "  kernel: 0=custom-tile  1=cuBLAS\n"
            "  prec:   0=FP64  1=FP32  2=TF32  3=FP16+TC(cuBLAS only)\n",
            argv[0]);
        MPI_Finalize(); 
        return 1;
    }

    int N = atoi(argv[1]);
    int pr = atoi(argv[2]);
    int pc = atoi(argv[3]);
    int kernel = atoi(argv[4]);
    int prec = (argc>=6) ? atoi(argv[5]) : PREC_FP64;
    const char *ckpt = (argc>=7) ? argv[6] : NULL;

    if (prec == PREC_FP16 && kernel == KERN_CUSTOM) {
        if (!wr) fprintf(stderr,
            "[warn] FP16+TC unsupported in custom kernel -- using FP32\n");
        prec = PREC_FP32;
    }

    const char *knames[] = {"custom-tile","cuBLAS"};
    const char *pnames[] = {"FP64","FP32","TF32","FP16+TC"};

    if (pr*pc != ws) {
        if (!wr) fprintf(stderr,"pr*pc must equal total MPI ranks\n");
        MPI_Finalize();
        return 1;
    }

    int mr = wr/pc,  mc = wr%pc;
    MPI_Comm rc, cc;
    MPI_Comm_split(MPI_COMM_WORLD, mr, mc, &rc);
    MPI_Comm_split(MPI_COMM_WORLD, mc, mr, &cc);

    int lr = N/pr, lc = N/pc;

    double *lA = (double*)malloc(lr*N*sizeof(double));
    double *lB = (double*)malloc(N*lc*sizeof(double));
    double *lC = (double*)calloc(lr*lc, sizeof(double));
    double *pA = (double*)malloc(lr*PANEL_B*sizeof(double));
    double *pB = (double*)malloc(PANEL_B*lc*sizeof(double));

    srand48(wr+1);
    for (int i=0; i<lr*N; i++) lA[i]=drand48();
    for (int i=0; i<N*lc; i++) lB[i]=drand48();

    int local_rank = 0;
    char *lr_env = getenv("SLURM_LOCALID");
    
    if (lr_env) {
        local_rank = atoi(lr_env);
    } else {
        MPI_Comm local_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, wr, MPI_INFO_NULL, &local_comm);
        MPI_Comm_rank(local_comm, &local_rank);
        MPI_Comm_free(&local_comm);
    }

    int ng;
    CUDA_CHECK(cudaGetDeviceCount(&ng));
    if (ng > 0) {
        CUDA_CHECK(cudaSetDevice(local_rank % ng));
    } else {
        if (wr == 0) fprintf(stderr, "Error: No GPUs detected on node!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int pAn=lr*PANEL_B, pBn=PANEL_B*lc, pCn=lr*lc;

    double *dpA64, *dpB64, *dC64;
    CUDA_CHECK(cudaMalloc(&dpA64, pAn*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dpB64, pBn*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dC64, pCn*sizeof(double)));
    CUDA_CHECK(cudaMemset(dC64, 0, pCn*sizeof(double)));

    float *dpA32=NULL, *dpB32=NULL, *dC32=NULL;
    if (prec != PREC_FP64) {
        CUDA_CHECK(cudaMalloc(&dpA32, pAn*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dpB32, pBn*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC32, pCn*sizeof(float)));
        CUDA_CHECK(cudaMemset(dC32, 0, pCn*sizeof(float)));
    }

    __half *dpAh=NULL, *dpBh=NULL;
    if (prec == PREC_FP16) {
        CUDA_CHECK(cudaMalloc(&dpAh, pAn*sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&dpBh, pBn*sizeof(__half)));
    }

    cublasHandle_t hnd;
    CUBLAS_CHECK(cublasCreate(&hnd));

    Timers tm = {0};
    int steps = (N+PANEL_B-1)/PANEL_B;
    MPI_Barrier(MPI_COMM_WORLD);
    uint64_t tts = clock_now();

    for (int s=0; s<steps; s++) {
        int ps = s*PANEL_B;
        int pw = (ps+PANEL_B<=N) ? PANEL_B : (N-ps);
        int bcr = s%pc;
        int brr = s%pr;
        int T = 256;

        uint64_t t0 = clock_now();
        if (mc==bcr)
            for (int r=0; r<lr; r++)
                memcpy(pA+r*pw, lA+r*N+ps, pw*sizeof(double));
        MPI_Bcast(pA, lr*pw, MPI_DOUBLE, bcr, rc);
        tm.bcast_a += clock_now()-t0;

        t0 = clock_now();
        if (mr==brr)
            memcpy(pB, lB+ps*lc, pw*lc*sizeof(double));
        MPI_Bcast(pB, pw*lc, MPI_DOUBLE, brr, cc);
        tm.bcast_b += clock_now()-t0;

        t0 = clock_now();
        CUDA_CHECK(cudaMemcpy(dpA64, pA, lr*pw*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dpB64, pB, pw*lc*sizeof(double), cudaMemcpyHostToDevice));
        if (prec==PREC_FP16) {
            d2h_cast<<<(lr*pw+T-1)/T,T>>>(dpA64, dpAh, lr*pw);
            d2h_cast<<<(pw*lc+T-1)/T,T>>>(dpB64, dpBh, pw*lc);
        } else if (prec!=PREC_FP64) {
            d2f_cast<<<(lr*pw+T-1)/T,T>>>(dpA64, dpA32, lr*pw);
            d2f_cast<<<(pw*lc+T-1)/T,T>>>(dpB64, dpB32, pw*lc);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        tm.h2d += clock_now()-t0;

        t0 = clock_now();
        local_gemm(hnd, kernel, prec, lr, pw, lc, dpA64, dpB64, dC64, dpA32, dpB32, dC32, dpAh,  dpBh);
        CUDA_CHECK(cudaDeviceSynchronize());
        tm.compute += clock_now()-t0;

        if (ckpt && ((s+1)%CKPT_FREQ==0 || s==steps-1)) {
            t0 = clock_now();
            if (prec==PREC_FP64) {
                CUDA_CHECK(cudaMemcpy(lC, dC64, pCn*sizeof(double), cudaMemcpyDeviceToHost));
            } else {
                f2d_cast<<<(pCn+T-1)/T,T>>>(dC32, dC64, pCn);
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(lC, dC64, pCn*sizeof(double), cudaMemcpyDeviceToHost));
            }
            tm.d2h += clock_now()-t0;
            char fn[512]; snprintf(fn,sizeof(fn),"%s_step%d.bin",ckpt,s);
            ckpt_write(fn, lC, lr, lc, N, mr, mc, &tm.io);
        }
    }

    {
        int T=256; uint64_t t0=clock_now();
        if (prec==PREC_FP64) {
            CUDA_CHECK(cudaMemcpy(lC, dC64, pCn*sizeof(double), cudaMemcpyDeviceToHost));
        } else {
            f2d_cast<<<(pCn+T-1)/T,T>>>(dC32, dC64, pCn);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(lC, dC64, pCn*sizeof(double), cudaMemcpyDeviceToHost));
        }
        tm.d2h += clock_now()-t0;
    }

    tm.total = clock_now()-tts;

    Timers g;
    MPI_Reduce(&tm.bcast_a,&g.bcast_a,1,MPI_UINT64_T,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&tm.bcast_b,&g.bcast_b,1,MPI_UINT64_T,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&tm.h2d,&g.h2d,1,MPI_UINT64_T,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&tm.compute,&g.compute,1,MPI_UINT64_T,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&tm.d2h,&g.d2h,1,MPI_UINT64_T,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&tm.io,&g.io,1,MPI_UINT64_T,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&tm.total,&g.total,1,MPI_UINT64_T,MPI_MAX,0,MPI_COMM_WORLD);

    if (!wr) {
        double wall = cyc2s(g.total);
        double gflops = (2.0*(double)N*(double)N*(double)N)/(wall*1e9);
        printf("=== Parallel Matrix Multiplication Results ===\n");
        printf("N=%d Ranks=%d Grid=%dx%d Kernel=%s Precision=%s\n",
               N,ws,pr,pc,knames[kernel],pnames[prec]);
        printf("Wall time: %.4f s\n", wall);
        printf("GFLOPS: %.2f\n",   gflops);
        printf("--- Breakdown (%%wall-time, max across ranks) ---\n");
        printf("  Bcast A: %.2f%%\n", 100.0*cyc2s(g.bcast_a)/wall);
        printf("  Bcast B: %.2f%%\n", 100.0*cyc2s(g.bcast_b)/wall);
        printf("  H->D: %.2f%%\n", 100.0*cyc2s(g.h2d)/wall);
        printf("  GPU compute: %.2f%%\n", 100.0*cyc2s(g.compute)/wall);
        printf("  D->H: %.2f%%\n", 100.0*cyc2s(g.d2h)/wall);
        if (ckpt)
        printf("  MPI-IO ckpt: %.2f%%\n", 100.0*cyc2s(g.io)/wall);
        printf("==============================================\n");
        fflush(stdout);
    }

    cublasDestroy(hnd);
    cudaFree(dpA64); cudaFree(dpB64); cudaFree(dC64);
    if (dpA32) cudaFree(dpA32);
    if (dpB32) cudaFree(dpB32);
    if (dC32) cudaFree(dC32);
    if (dpAh) cudaFree(dpAh);
    if (dpBh) cudaFree(dpBh);
    free(lA);
    free(lB);
    free(lC);
    free(pA);
    free(pB);
    MPI_Comm_free(&rc);
    MPI_Comm_free(&cc);
    MPI_Finalize();
    return 0;
}