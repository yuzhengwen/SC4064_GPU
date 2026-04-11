#include "gpu_bsearch.h"
#include "cuda_check.h"
#include <cstdio>
#include <climits>
#include <algorithm>
using std::min;

/* ================================================================== */
/*  Configuration (must match gpu_smem.cu)                             */
/* ================================================================== */
#define TILE_SIZE    512
#define SMEM_THREADS (TILE_SIZE / 2)

/* ================================================================== */
/*  Phase 1 kernel — identical to gpu_smem (sort tiles in smem)        */
/* ================================================================== */
__global__ void bsearch_local_sort_k(const int *__restrict__ src,
                                      int       *__restrict__ dst,
                                      int n)
{
    __shared__ int s[2][TILE_SIZE];

    int block_start = blockIdx.x * TILE_SIZE;
    int tid         = threadIdx.x;

    int i0 = block_start + tid;
    int i1 = block_start + tid + TILE_SIZE / 2;
    s[0][tid]               = (i0 < n) ? src[i0] : INT_MAX;
    s[0][tid + TILE_SIZE/2] = (i1 < n) ? src[i1] : INT_MAX;
    __syncthreads();

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

    if (i0 < n) dst[i0] = s[cur][tid];
    if (i1 < n) dst[i1] = s[cur][tid + TILE_SIZE / 2];
}

/* ================================================================== */
/*  Co-rank (binary search on merge boundary)                          */
/*                                                                     */
/*  Returns i* such that the first k elements of merge(A[0..m),        */
/*  B[0..p)) come from A[0..i*) and B[0..k-i*).                       */
/*                                                                     */
/*  Property: A[i*-1] <= B[k-i*]  (or boundary)                       */
/*            B[k-i*-1] < A[i*]   (or boundary)                       */
/*  This makes equal keys stable: A's copy precedes B's copy.          */
/* ================================================================== */
__device__ static long long co_rank(long long k,
                                     const int *A, long long m,
                                     const int *B, long long p)
{
    long long lo = (k > p) ? (k - p) : 0LL;
    long long hi = (k < m) ? k       : m;

    while (lo < hi) {
        long long i = lo + (hi - lo) / 2;
        long long j = k - i;
        /*
         * Advance lo when A[i] still belongs before B[j-1]
         * (strict < keeps equal elements from A before equal from B)
         */
        if (i < m && j > 0 && A[i] < B[j - 1])
            lo = i + 1;
        else
            hi = i;
    }
    return lo;
}

/* ================================================================== */
/*  Phase 2 kernel — parallel merge, one thread per output element     */
/*                                                                     */
/*  Grid size = ceil(n / THREADS).  Thread gid writes dst[gid].        */
/*  The co-rank binary search costs O(log width) per thread but all    */
/*  n threads run concurrently — no sequential merge bottleneck.       */
/* ================================================================== */
__global__ void bsearch_merge_k(const int *__restrict__ src,
                                  int       *__restrict__ dst,
                                  long long n, long long width)
{
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    /* Which merge segment does this output position belong to? */
    long long seg   = gid / (2LL * width);
    long long left  = seg * 2LL * width;
    long long mid_p = min(left + width,        n);   /* start of right half */
    long long right = min(left + 2LL * width,  n);   /* exclusive end       */

    /* Local position within this merge's output */
    long long local_pos = gid - left;   /* always < right - left (proved in header) */

    long long m = mid_p - left;         /* length of left  run */
    long long p = right - mid_p;        /* length of right run */

    /*
     * Use co-rank to split output[local_pos] between the two runs:
     *   i elements come from the left run  (A = src + left)
     *   j elements come from the right run (B = src + mid_p)
     */
    long long i = co_rank(local_pos, src + left, m, src + mid_p, p);
    long long j = local_pos - i;

    /* Pick the smaller of A[i] and B[j] */
    if (i < m && (j >= p || src[left + i] <= src[mid_p + j]))
        dst[gid] = src[left + i];
    else
        dst[gid] = src[mid_p + j];
}

/* ================================================================== */
/*  Host launcher                                                       */
/* ================================================================== */
float run_gpu_bsearch(const int *h_src, int *h_dst, int n)
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

        bsearch_local_sort_k<<<num_blocks, SMEM_THREADS>>>(d_a, d_b, n);

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

        int *tmp = d_a; d_a = d_b; d_b = tmp;
    }

    /* ---- Phase 2: parallel co-rank merges from TILE_SIZE upward ---- */
    printf("    %-5s  %-14s  %-12s  %-12s  %-12s  %-10s\n",
           "Pass", "Width", "Merges", "Time(ms)", "Elem/s", "BW(GB/s)");
    printf("    %s\n",
           "-----------------------------------------------------------------------");

    int pass = 1;
    for (long long width = TILE_SIZE; width < (long long)n; width *= 2) {
        long long num_merges = ((long long)n + 2LL * width - 1) / (2LL * width);
        /* One thread per output element */
        int blocks = (int)(((long long)n + THREADS - 1) / THREADS);

        cudaEvent_t ps, pe;
        CUDA_CHECK(cudaEventCreate(&ps));
        CUDA_CHECK(cudaEventCreate(&pe));
        CUDA_CHECK(cudaEventRecord(ps));

        bsearch_merge_k<<<blocks, THREADS>>>(d_a, d_b, (long long)n, width);

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
