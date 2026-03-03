#include <stdio.h>
#include <climits>
#include "gpu_smem.h"
#include "cuda_check.h"

#define SMEM_SIZE 1024

__global__ void smem_local_sort_kernel(int *src, int *dst, int n)
{
    __shared__ int s_buf[2][SMEM_SIZE];
    int block_start = blockIdx.x * SMEM_SIZE;
    int tid         = threadIdx.x;
    int idx0 = block_start + tid;
    int idx1 = block_start + tid + SMEM_SIZE / 2;
    s_buf[0][tid]               = (idx0 < n) ? src[idx0] : INT_MAX;
    s_buf[0][tid + SMEM_SIZE/2] = (idx1 < n) ? src[idx1] : INT_MAX;
    __syncthreads();
    int cur = 0;
    for (int width = 1; width < SMEM_SIZE; width *= 2) {
        int nxt = 1 - cur;
        int num_merges = SMEM_SIZE / (2 * width);
        if (tid < num_merges) {
            int left  = tid * 2 * width;
            int mid   = left + width;
            int right = left + 2 * width;
            int i = left, j = mid, k = left;
            while (i < mid && j < right)
                s_buf[nxt][k++] = (s_buf[cur][i] <= s_buf[cur][j]) ? s_buf[cur][i++] : s_buf[cur][j++];
            while (i < mid)   s_buf[nxt][k++] = s_buf[cur][i++];
            while (j < right) s_buf[nxt][k++] = s_buf[cur][j++];
        }
        cur = nxt;
        __syncthreads();
    }
    if (idx0 < n) dst[idx0] = s_buf[cur][tid];
    if (idx1 < n) dst[idx1] = s_buf[cur][tid + SMEM_SIZE/2];
}

__global__ void smem_global_merge_kernel(int *src, int *dst, long long n, long long width)
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

int *gpu_mergeSort_smem(int *d_data, int *d_temp, int n)
{
    int num_blocks     = (n + SMEM_SIZE - 1) / SMEM_SIZE;
    int phase1_threads = SMEM_SIZE / 2;
    smem_local_sort_kernel<<<num_blocks, phase1_threads>>>(d_data, d_temp, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int *cur = d_temp;
    int *nxt = d_data;
    const int threads = 256;
    for (long long width = SMEM_SIZE; width < (long long)n; width *= 2) {
        long long num_merges = ((long long)n + 2LL * width - 1) / (2LL * width);
        int blocks = (int)((num_merges + threads - 1) / threads);
        smem_global_merge_kernel<<<blocks, threads>>>(cur, nxt, (long long)n, width);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        int *tmp = cur; cur = nxt; nxt = tmp;
    }
    return cur;
}
