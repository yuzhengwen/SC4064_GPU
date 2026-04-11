#include "gpu_smem.h"
#include "cuda_check.h"
#include <cstdio>
#include <climits>
#include <algorithm>
using std::min;

/* ================================================================== */
/*  Configuration                                                       */
/* ================================================================== */
#define TILE_SIZE 512          /* elements per smem tile (power of 2) */
#define SMEM_THREADS (TILE_SIZE / 2)  /* each thread loads 2 elements */

/* ================================================================== */
/*  Phase 1 kernel — sort one tile entirely in shared memory           */
/*                                                                     */
/*  Each block sorts TILE_SIZE elements.  We double-buffer the tile    */
/*  and run log2(TILE_SIZE) merge passes on-chip.                      */
/*  Thread count per block = TILE_SIZE / 2; each thread owns 2 elems. */
/* ================================================================== */
__global__ void smem_local_sort_k(const int *__restrict__ src,
                                   int       *__restrict__ dst,
                                   int n)
{
    __shared__ int s[2][TILE_SIZE];

    int block_start = blockIdx.x * TILE_SIZE;
    int tid         = threadIdx.x;            /* 0 .. TILE_SIZE/2 - 1 */

    /* Load two elements per thread; pad out-of-range positions */
    int i0 = block_start + tid;
    int i1 = block_start + tid + TILE_SIZE / 2;
    s[0][tid]               = (i0 < n) ? src[i0] : INT_MAX;
    s[0][tid + TILE_SIZE/2] = (i1 < n) ? src[i1] : INT_MAX;
    __syncthreads();

    /*
     * Bottom-up merge passes inside shared memory.
     * At each pass, num_merges threads are active.  Each active thread
     * merges two adjacent sorted runs of length `width` into `s[nxt]`.
     * Because num_merges * 2 * width == TILE_SIZE, every element is
     * written exactly once per pass — no stale data issue.
     */
    int cur = 0;
    for (int width = 1; width < TILE_SIZE; width *= 2) {
        int nxt        = 1 - cur;
        int num_merges = TILE_SIZE / (2 * width);

        if (tid < num_merges) {
            int left  = tid * 2 * width;
            int mid   = left + width;
            int right = left + 2 * width;
            int i = left, j = mid, k = left;
            while (i < mid && j < right)
                s[nxt][k++] = (s[cur][i] <= s[cur][j]) ? s[cur][i++] : s[cur][j++];
            while (i < mid)   s[nxt][k++] = s[cur][i++];
            while (j < right) s[nxt][k++] = s[cur][j++];
        }
        cur = nxt;
        __syncthreads();
    }

    /* Write sorted tile back to global memory (ignore INT_MAX sentinels) */
    if (i0 < n) dst[i0] = s[cur][tid];
    if (i1 < n) dst[i1] = s[cur][tid + TILE_SIZE / 2];
}

/* ================================================================== */
/*  Phase 2 kernel — global-memory merge (same as naive)               */
/*  Starts at width = TILE_SIZE, so TILE_SIZE passes are already done. */
/* ================================================================== */
__global__ void smem_global_merge_k(const int *__restrict__ src,
                                     int       *__restrict__ dst,
                                     long long n, long long width)
{
    long long tid   = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long left  = tid * 2LL * width;
    if (left >= n) return;

    long long mid   = min(left + width,       n);
    long long right = min(left + 2LL * width, n);

    if (mid >= right) {
        for (long long x = left; x < right; x++) dst[x] = src[x];
        return;
    }

    long long i = left, j = mid, k = left;
    while (i < mid && j < right)
        dst[k++] = (src[i] <= src[j]) ? src[i++] : src[j++];
    while (i < mid)   dst[k++] = src[i++];
    while (j < right) dst[k++] = src[j++];
}

/* ================================================================== */
/*  Host launcher                                                       */
/* ================================================================== */
float run_gpu_smem(const int *h_src, int *h_dst, int n)
{
    const int THREADS = 256;
    size_t bytes = (size_t)n * sizeof(int);

    int *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_src, bytes, cudaMemcpyHostToDevice));

    float total_ms = 0.0f;

    /* ---- Phase 1: sort tiles in shared memory ---- */
    {
        int num_blocks = (n + TILE_SIZE - 1) / TILE_SIZE;
        cudaEvent_t ps, pe;
        CUDA_CHECK(cudaEventCreate(&ps));
        CUDA_CHECK(cudaEventCreate(&pe));
        CUDA_CHECK(cudaEventRecord(ps));

        smem_local_sort_k<<<num_blocks, SMEM_THREADS>>>(d_a, d_b, n);

        CUDA_CHECK(cudaEventRecord(pe));
        CUDA_CHECK(cudaEventSynchronize(pe));
        CUDA_CHECK(cudaGetLastError());

        float pms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&pms, ps, pe));
        total_ms += pms;

        double tp = n / (pms / 1000.0);
        double bw = 2.0 * bytes / (pms / 1000.0) / 1e9;
        printf("    %-5s  %-14s  %-12d  %-12.4f  %-12.3e  %-10.3f\n",
               "P1", "smem-tile", num_blocks, pms, tp, bw);

        CUDA_CHECK(cudaEventDestroy(ps));
        CUDA_CHECK(cudaEventDestroy(pe));

        /* d_b now holds TILE_SIZE-sorted segments */
        int *tmp = d_a; d_a = d_b; d_b = tmp;
    }

    /* ---- Phase 2: global merges from TILE_SIZE upward ---- */
    printf("    %-5s  %-14s  %-12s  %-12s  %-12s  %-10s\n",
           "Pass", "Width", "Merges", "Time(ms)", "Elem/s", "BW(GB/s)");
    printf("    %s\n",
           "-----------------------------------------------------------------------");

    int pass = 1;
    for (long long width = TILE_SIZE; width < (long long)n; width *= 2) {
        long long num_merges = ((long long)n + 2LL * width - 1) / (2LL * width);
        int blocks = (int)((num_merges + THREADS - 1) / THREADS);

        cudaEvent_t ps, pe;
        CUDA_CHECK(cudaEventCreate(&ps));
        CUDA_CHECK(cudaEventCreate(&pe));
        CUDA_CHECK(cudaEventRecord(ps));

        smem_global_merge_k<<<blocks, THREADS>>>(d_a, d_b, (long long)n, width);

        CUDA_CHECK(cudaEventRecord(pe));
        CUDA_CHECK(cudaEventSynchronize(pe));
        CUDA_CHECK(cudaGetLastError());

        float pms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&pms, ps, pe));
        total_ms += pms;

        double tp = n / (pms / 1000.0);
        double bw = 2.0 * bytes / (pms / 1000.0) / 1e9;
        printf("    %-5d  %-14lld  %-12lld  %-12.4f  %-12.3e  %-10.3f\n",
               pass, width, num_merges, pms, tp, bw);

        CUDA_CHECK(cudaEventDestroy(ps));
        CUDA_CHECK(cudaEventDestroy(pe));

        int *tmp = d_a; d_a = d_b; d_b = tmp;
        pass++;
    }

    printf("    %s\n",
           "-----------------------------------------------------------------------");
    printf("    Total kernel time: %.4f ms\n", total_ms);

    CUDA_CHECK(cudaMemcpy(h_dst, d_a, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    return total_ms;
}
