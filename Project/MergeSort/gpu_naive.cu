#include <stdio.h>
#include "gpu_naive.h"
#include "cuda_check.h"

__global__ void gpu_merge_kernel(int *src, int *dst, long long n, long long width)
{
    long long tid  = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long left = tid * 2LL * width;
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

int *gpu_mergeSort_naive(int *d_data, int *d_temp, int n)
{
    const int threads = 256;
    int pass = 1;
    float total_ms = 0.0f;

    printf("\n  %-6s  %-18s  %-14s  %-10s\n", "Pass", "Width", "Merges", "Time (ms)");
    printf("  ---------------------------------------------------\n");

    for (long long width = 1; width < (long long)n; width *= 2) {
        long long num_merges = ((long long)n + 2LL * width - 1) / (2LL * width);
        int blocks = (int)((num_merges + threads - 1) / threads);

        cudaEvent_t ps, pe;
        CUDA_CHECK(cudaEventCreate(&ps));
        CUDA_CHECK(cudaEventCreate(&pe));
        CUDA_CHECK(cudaEventRecord(ps));

        gpu_merge_kernel<<<blocks, threads>>>(d_data, d_temp, (long long)n, width);

        CUDA_CHECK(cudaEventRecord(pe));
        CUDA_CHECK(cudaEventSynchronize(pe));
        CUDA_CHECK(cudaGetLastError());

        float pass_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&pass_ms, ps, pe));
        total_ms += pass_ms;

        printf("  %-6d  %-18lld  %-14lld  %.3f ms\n", pass, width, num_merges, pass_ms);

        CUDA_CHECK(cudaEventDestroy(ps));
        CUDA_CHECK(cudaEventDestroy(pe));

        int *tmp = d_data; d_data = d_temp; d_temp = tmp;
        pass++;
    }

    printf("  ---------------------------------------------------\n");
    printf("  Total naive GPU time: %.3f ms\n\n", total_ms);

    return d_data;
}
