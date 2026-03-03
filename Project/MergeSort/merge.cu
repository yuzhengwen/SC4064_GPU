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
    int n1 = mid - left + 1;
    int n2 = right - mid;
    int *L = (int *)malloc(n1 * sizeof(int));
    int *R = (int *)malloc(n2 * sizeof(int));
    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int j = 0; j < n2; j++) R[j] = arr[mid + 1 + j];
    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2)
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
    free(L);
    free(R);
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
// 2. Naive GPU merge sort (global memory only, one pass per kernel launch)
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
    for (long long width = 1; width < (long long)n; width *= 2) {
        long long num_merges = ((long long)n + 2LL * width - 1) / (2LL * width);
        int blocks = (int)((num_merges + threads - 1) / threads);
        gpu_merge_kernel<<<blocks, threads>>>(d_data, d_temp, (long long)n, width);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        int *tmp = d_data; d_data = d_temp; d_temp = tmp;
    }
    return d_data;
}

// =============================================================================
// 3. Shared memory tiled GPU merge sort
//
// Phase 1 — block-level sort entirely in shared memory:
//   Each block loads SMEM_SIZE elements from global memory into shared memory,
//   then runs all log2(SMEM_SIZE) merge passes locally without any global
//   memory traffic between passes. Only 1 global read + 1 global write per
//   block, regardless of how many local passes run.
//
// Phase 2 — large merges in global memory:
//   After Phase 1 every chunk of SMEM_SIZE is sorted. The remaining
//   log2(N/SMEM_SIZE) passes are run as standard global-memory merges,
//   but there are far fewer of them (15 instead of 25 for N=32M, SMEM_SIZE=1024).
// =============================================================================

#define SMEM_SIZE 1024  // elements per block; must be power of 2 and <= 1024

// Phase 1 kernel: sort a SMEM_SIZE-element chunk entirely in shared memory
__global__ void smem_local_sort_kernel(int *src, int *dst, int n)
{
    __shared__ int smem[SMEM_SIZE];
    __shared__ int stmp[SMEM_SIZE];

    int block_start = blockIdx.x * SMEM_SIZE;
    int tid         = threadIdx.x;  // 0..SMEM_SIZE-1

    // Load from global memory into shared memory
    int global_idx = block_start + tid;
    smem[tid] = (global_idx < n) ? src[global_idx] : INT_MAX; // pad with INT_MAX
    __syncthreads();

    // Run all log2(SMEM_SIZE) merge passes entirely inside shared memory.
    // Each thread handles one merge pair per pass.
    // smem is input, stmp is output; ping-pong between them.
    int *s_in  = smem;
    int *s_out = stmp;

    for (int width = 1; width < SMEM_SIZE; width *= 2) {
        // This thread is responsible for merge chunk (tid / (2*width))
        int chunk     = tid / (2 * width);
        int local_left  = chunk * 2 * width;
        int local_mid   = local_left + width;
        int local_right = local_left + 2 * width;

        // Position within the merge output
        int pos = tid - local_left;

        // Binary search to find where this thread's output element comes from.
        // Each thread writes exactly one output element using the diagonal
        // intersection technique: given output position pos, find split (p, q)
        // such that p elements come from left half and q=pos-p from right half.
        int p_lo = max(0, pos - width);
        int p_hi = min(pos, width);

        while (p_lo < p_hi) {
            int p_mid = (p_lo + p_hi) / 2;
            int q_mid = pos - p_mid - 1;
            if (s_in[local_left + p_mid] > s_in[local_mid + q_mid])
                p_hi = p_mid;
            else
                p_lo = p_mid + 1;
        }

        int p = p_lo;
        int q = pos - p;

        // Pick the correct element
        bool take_left;
        if (p >= width)       take_left = false;
        else if (q >= width)  take_left = true;
        else take_left = (s_in[local_left + p] <= s_in[local_mid + q]);

        s_out[local_left + pos] = take_left
            ? s_in[local_left  + p]
            : s_in[local_mid   + q];

        __syncthreads();

        // Swap ping-pong buffers
        int *tmp = s_in; s_in = s_out; s_out = tmp;
    }

    // Write sorted chunk back to global memory (only real elements)
    if (global_idx < n)
        dst[global_idx] = s_in[tid];
}

// Phase 2 kernel: same as naive global merge, used for widths >= SMEM_SIZE
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
    const int threads = SMEM_SIZE; // one thread per element in shared memory

    // Phase 1: sort each SMEM_SIZE block locally in shared memory
    int num_blocks = (n + SMEM_SIZE - 1) / SMEM_SIZE;
    smem_local_sort_kernel<<<num_blocks, threads>>>(d_data, d_temp, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // After Phase 1, sorted chunks of size SMEM_SIZE are in d_temp.
    // Ping-pong: d_temp is now input for Phase 2.
    int *cur = d_temp;
    int *nxt = d_data;

    // Phase 2: merge sorted chunks using global memory passes
    // Start at width=SMEM_SIZE (chunks are already sorted up to this width)
    const int global_threads = 256;
    for (long long width = SMEM_SIZE; width < (long long)n; width *= 2) {
        long long num_merges = ((long long)n + 2LL * width - 1) / (2LL * width);
        int blocks = (int)((num_merges + global_threads - 1) / global_threads);
        smem_global_merge_kernel<<<blocks, global_threads>>>(cur, nxt, (long long)n, width);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        int *tmp = cur; cur = nxt; nxt = tmp;
    }

    return cur;
}

// =============================================================================
// 4. Thrust sort
// =============================================================================
void thrust_sort(int *h_src, int *h_dst, int n)
{
    thrust::device_vector<int> d_vec(h_src, h_src + n);
    thrust::sort(d_vec.begin(), d_vec.end());
    thrust::copy(d_vec.begin(), d_vec.end(), h_dst);
}

// Thrust timed separately so we can wrap CUDA events around it
float thrust_sort_timed(int *h_src, int *h_dst, int n)
{
    // Upload
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

void print_result(const char *label, double cpu_time,
                  double this_time_ms, int correct)
{
    double speedup = (cpu_time * 1000.0) / this_time_ms;
    printf("  %-30s %8.2f ms   speedup: %5.2fx   %s\n",
           label, this_time_ms,
           speedup,
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
    printf("  N = %d elements  (%.0f MB per array)\n",
           N, bytes / 1e6);
    printf("=======================================================\n\n");

    int *h_original = (int *)malloc(bytes);
    int *h_cpu      = (int *)malloc(bytes);
    int *h_gpu      = (int *)malloc(bytes);

    srand(42);
    for (int i = 0; i < N; i++)
        h_original[i] = rand() % 1000000;

    // ------------------------------------------------------------------
    // CPU baseline
    // ------------------------------------------------------------------
    memcpy(h_cpu, h_original, bytes);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    cpu_mergeSort(h_cpu, 0, N - 1);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double cpu_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf("[CPU] Sequential merge sort: %.4f seconds\n\n", cpu_time);
    printf("%-32s %10s   %14s   %s\n", "Variant", "Time (ms)", "vs CPU", "Correct?");
    printf("---------------------------------------------------------------\n");

    // ------------------------------------------------------------------
    // Naive GPU
    // ------------------------------------------------------------------
    int *d_data, *d_temp;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMalloc(&d_temp, bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_original, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_temp, h_original, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start));
    int *d_result = gpu_mergeSort_naive(d_data, d_temp, N);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float naive_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&naive_ms, ev_start, ev_stop));
    CUDA_CHECK(cudaMemcpy(h_gpu, d_result, bytes, cudaMemcpyDeviceToHost));
    print_result("Naive GPU (global mem only)", cpu_time, naive_ms,
                 verify(h_cpu, h_gpu, N));

    // ------------------------------------------------------------------
    // Shared memory tiled GPU
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(d_data, h_original, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_temp, h_original, bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(ev_start));
    d_result = gpu_mergeSort_smem(d_data, d_temp, N);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float smem_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&smem_ms, ev_start, ev_stop));
    CUDA_CHECK(cudaMemcpy(h_gpu, d_result, bytes, cudaMemcpyDeviceToHost));
    print_result("Shared memory tiled GPU", cpu_time, smem_ms,
                 verify(h_cpu, h_gpu, N));

    // ------------------------------------------------------------------
    // Thrust
    // ------------------------------------------------------------------
    float thrust_ms = thrust_sort_timed(h_original, h_gpu, N);
    print_result("Thrust sort", cpu_time, thrust_ms,
                 verify(h_cpu, h_gpu, N));

    printf("---------------------------------------------------------------\n\n");
    printf("CPU baseline: %.2f ms\n", cpu_time * 1000.0);

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_temp));
    free(h_original);
    free(h_cpu);
    free(h_gpu);
    return 0;
}
