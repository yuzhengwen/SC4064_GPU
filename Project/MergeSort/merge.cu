// Merge sort benchmark: CPU vs GPU (naive) vs GPU (shared memory) vs Thrust
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#define CUDA_CHECK(call)                                          \
    do                                                            \
    {                                                             \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA error: %s at %s:%d\n",          \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(err);                                            \
        }                                                         \
    } while (0)

// =============================================================================
// 1. CPU sequential merge sort
// =============================================================================
void cpu_merge(int *arr, int left, int mid, int right)
{
    int n1 = mid - left + 1, n2 = right - mid;
    int *L = (int *)malloc(n1 * sizeof(int));
    int *R = (int *)malloc(n2 * sizeof(int));
    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int j = 0; j < n2; j++) R[j] = arr[mid + 1 + j];
    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2)
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
    free(L); free(R);
}
void cpu_mergeSort(int *arr, int left, int right)
{
    if (left < right) {
        int mid = left + (right - left) / 2;
        cpu_mergeSort(arr, left, mid);
        cpu_mergeSort(arr, mid + 1, right);
        cpu_merge(arr, left, mid, right);
    }
}

// =============================================================================
// 2. Naive GPU merge sort (global memory only)
// =============================================================================
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

    printf("\n  %-6s  %-18s  %-10s  %-10s\n",
           "Pass", "Width", "Threads", "Time (ms)");
    printf("  -----------------------------------------------\n");

    for (long long width = 1; width < (long long)n; width *= 2) {
        long long num_merges = ((long long)n + 2LL * width - 1) / (2LL * width);
        int blocks = (int)((num_merges + threads - 1) / threads);

        // Create per-pass CUDA events
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

        printf("  %-6d  %-18lld  %-10lld  %.3f ms\n",
               pass, width, num_merges, pass_ms);

        CUDA_CHECK(cudaEventDestroy(ps));
        CUDA_CHECK(cudaEventDestroy(pe));

        int *tmp = d_data; d_data = d_temp; d_temp = tmp;
        pass++;
    }

    printf("  -----------------------------------------------\n");
    printf("  Total GPU time: %.3f ms\n\n", total_ms);

    return d_data;
}

// =============================================================================
// 3. Shared memory tiled GPU merge sort
//
// The key insight: instead of a binary search per thread (which is slow),
// we use a single warp (32 threads) to cooperatively merge two sorted halves
// inside shared memory using a simple sequential merge.
//
// SMEM_SIZE = 1024 elements per block.
// Each block handles ONE chunk of 1024 elements.
//
// Phase 1 (smem_local_sort_kernel):
//   - Load 1024 elements into shared memory (1 global read)
//   - Run log2(1024) = 10 merge passes entirely in smem using __syncthreads()
//   - Each pass: thread i owns merge chunk i, does a simple sequential merge
//     of its two sub-arrays within smem into a temp smem buffer
//   - Write sorted 1024-element chunk back (1 global write)
//   -> Saves 9 global memory round trips per 1024-element chunk vs naive
//
// Phase 2 (smem_global_merge_kernel):
//   - Standard global memory merge for widths >= SMEM_SIZE
//   - Only log2(N/1024) passes needed instead of log2(N)
//   - For N=32M: 15 passes instead of 25
// =============================================================================

#define SMEM_SIZE 1024

// Phase 1: sort each SMEM_SIZE chunk entirely in shared memory.
// Block size = SMEM_SIZE/2 threads. Each thread owns one merge pair per pass.
__global__ void smem_local_sort_kernel(int *src, int *dst, int n)
{
    // Two shared memory buffers for ping-pong within the block
    __shared__ int s_buf[2][SMEM_SIZE];

    int block_start = blockIdx.x * SMEM_SIZE;
    int tid         = threadIdx.x;  // 0 .. SMEM_SIZE/2 - 1

    // Load two elements per thread (thread tid loads indices tid and tid + SMEM_SIZE/2)
    int idx0 = block_start + tid;
    int idx1 = block_start + tid + SMEM_SIZE / 2;
    s_buf[0][tid]               = (idx0 < n) ? src[idx0] : INT_MAX;
    s_buf[0][tid + SMEM_SIZE/2] = (idx1 < n) ? src[idx1] : INT_MAX;
    __syncthreads();

    int cur = 0; // which s_buf is the current input

    // Run all log2(SMEM_SIZE) merge passes inside shared memory
    for (int width = 1; width < SMEM_SIZE; width *= 2) {
        int nxt = 1 - cur;

        // Each thread is responsible for one merge operation.
        // Thread tid merges the pair at chunk index tid (relative to this width).
        // There are SMEM_SIZE / (2*width) merge ops per pass,
        // so threads with tid >= that count are idle this pass.
        int num_merges = SMEM_SIZE / (2 * width);
        if (tid < num_merges) {
            int left  = tid * 2 * width;
            int mid   = left + width;
            int right = left + 2 * width;

            // Simple sequential merge of s_buf[cur][left..mid-1]
            // and s_buf[cur][mid..right-1] into s_buf[nxt][left..right-1]
            int i = left, j = mid, k = left;
            while (i < mid && j < right)
                s_buf[nxt][k++] = (s_buf[cur][i] <= s_buf[cur][j])
                                    ? s_buf[cur][i++]
                                    : s_buf[cur][j++];
            while (i < mid)   s_buf[nxt][k++] = s_buf[cur][i++];
            while (j < right) s_buf[nxt][k++] = s_buf[cur][j++];
        }
        cur = nxt;
        __syncthreads();
    }

    // Write sorted chunk back to global memory (skip padded INT_MAX sentinels)
    if (idx0 < n) dst[idx0] = s_buf[cur][tid];
    if (idx1 < n) dst[idx1] = s_buf[cur][tid + SMEM_SIZE/2];
}

// Phase 2: global memory merges for widths >= SMEM_SIZE (same as naive)
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
    // Phase 1: sort each SMEM_SIZE block in shared memory
    int num_blocks   = (n + SMEM_SIZE - 1) / SMEM_SIZE;
    int phase1_threads = SMEM_SIZE / 2; // each thread handles 2 elements
    smem_local_sort_kernel<<<num_blocks, phase1_threads>>>(d_data, d_temp, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // After Phase 1, sorted chunks of SMEM_SIZE are in d_temp
    int *cur = d_temp;
    int *nxt = d_data;

    // Phase 2: global memory merges starting from width = SMEM_SIZE
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

// =============================================================================
// 4. Thrust sort
// =============================================================================
float thrust_sort_timed(int *h_src, int *h_dst, int n)
{
    int *d_arr;
    CUDA_CHECK(cudaMalloc(&d_arr, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_arr, h_src, (size_t)n * sizeof(int), cudaMemcpyHostToDevice));
    thrust::device_ptr<int> d_ptr(d_arr);

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start));
    thrust::sort(d_ptr, d_ptr + n);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    CUDA_CHECK(cudaMemcpy(h_dst, d_arr, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_arr));
    return ms;
}

// =============================================================================
// Helpers
// =============================================================================
int verify(int *ref, int *result, int n)
{
    for (int i = 0; i < n; i++)
        if (ref[i] != result[i]) return 0;
    return 1;
}
void print_result(const char *label, double cpu_ms, double this_ms, int correct)
{
    printf("  %-34s %8.2f ms   speedup: %6.2fx   %s\n",
           label, this_ms, cpu_ms / this_ms,
           correct ? "PASSED" : "FAILED");
}

// =============================================================================
// Main
// =============================================================================
int main()
{
    int N = 1 << 25;
    size_t bytes = (size_t)N * sizeof(int);

    printf("=======================================================\n");
    printf("  Merge Sort Benchmark\n");
    printf("  N = %d elements  (%.0f MB per array)\n", N, bytes / 1e6);
    printf("=======================================================\n\n");

    int *h_original = (int *)malloc(bytes);
    int *h_cpu      = (int *)malloc(bytes);
    int *h_gpu      = (int *)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; i++) h_original[i] = rand() % 1000000;

    // CPU baseline
    memcpy(h_cpu, h_original, bytes);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    cpu_mergeSort(h_cpu, 0, N - 1);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double cpu_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    double cpu_ms   = cpu_time * 1000.0;
    printf("[CPU] Sequential merge sort: %.4f seconds\n\n", cpu_time);
    printf("%-36s %10s   %14s   %s\n", "Variant", "Time (ms)", "vs CPU", "Correct?");
    printf("-----------------------------------------------------------------\n");

    int *d_data, *d_temp;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMalloc(&d_temp, bytes));
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // Naive GPU
    CUDA_CHECK(cudaMemcpy(d_data, h_original, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_temp, h_original, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ev_start));
    int *d_result = gpu_mergeSort_naive(d_data, d_temp, N);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float naive_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&naive_ms, ev_start, ev_stop));
    CUDA_CHECK(cudaMemcpy(h_gpu, d_result, bytes, cudaMemcpyDeviceToHost));
    print_result("Naive GPU (global mem only)", cpu_ms, naive_ms, verify(h_cpu, h_gpu, N));

    // Shared memory tiled GPU
    CUDA_CHECK(cudaMemcpy(d_data, h_original, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_temp, h_original, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ev_start));
    d_result = gpu_mergeSort_smem(d_data, d_temp, N);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float smem_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&smem_ms, ev_start, ev_stop));
    CUDA_CHECK(cudaMemcpy(h_gpu, d_result, bytes, cudaMemcpyDeviceToHost));
    print_result("Shared memory tiled GPU", cpu_ms, smem_ms, verify(h_cpu, h_gpu, N));

    // Thrust
    float thrust_ms = thrust_sort_timed(h_original, h_gpu, N);
    print_result("Thrust sort", cpu_ms, thrust_ms, verify(h_cpu, h_gpu, N));

    printf("-----------------------------------------------------------------\n\n");
    printf("CPU baseline: %.2f ms\n", cpu_ms);

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_temp));
    free(h_original); free(h_cpu); free(h_gpu);
    return 0;
}
