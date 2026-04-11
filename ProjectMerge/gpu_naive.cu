#include "gpu_naive.h"
#include "cuda_check.h"
#include <cstdio>
#include <algorithm>
using std::min;

/* ================================================================== */
/*  Kernel: one thread merges one pair of sorted runs                  */
/*  Each thread owns the full sequential merge of its segment.         */
/* ================================================================== */
__global__ void naive_merge_k(const int *__restrict__ src,
                               int       *__restrict__ dst,
                               long long n, long long width)
{
    long long tid   = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long left  = tid * 2LL * width;
    if (left >= n) return;

    long long mid   = min(left + width,        n);
    long long right = min(left + 2LL * width,  n);

    /* If there is no right half, just copy */
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
float run_gpu_naive(const int *h_src, int *h_dst, int n)
{
    const int THREADS = 256;
    size_t bytes = (size_t)n * sizeof(int);

    int *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_src, bytes, cudaMemcpyHostToDevice));

    printf("    %-5s  %-14s  %-12s  %-12s  %-12s  %-10s\n",
           "Pass", "Width", "Merges", "Time(ms)", "Elem/s", "BW(GB/s)");
    printf("    %s\n",
           "-----------------------------------------------------------------------");

    float total_ms = 0.0f;
    int   pass     = 1;

    for (long long width = 1; width < (long long)n; width *= 2) {
        long long num_merges = ((long long)n + 2LL * width - 1) / (2LL * width);
        int blocks = (int)((num_merges + THREADS - 1) / THREADS);

        cudaEvent_t ps, pe;
        CUDA_CHECK(cudaEventCreate(&ps));
        CUDA_CHECK(cudaEventCreate(&pe));
        CUDA_CHECK(cudaEventRecord(ps));

        naive_merge_k<<<blocks, THREADS>>>(d_a, d_b, (long long)n, width);

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

        /* Ping-pong buffers */
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
