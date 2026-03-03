// Parallel merge sort using CUDA
// Strategy: bottom-up iterative merge sort where each merge at each level
// is parallelized across GPU threads.

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Error checking macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error: %s at %s:%d\n",                         \
                    cudaGetErrorString(err), __FILE__, __LINE__);                 \
            exit(err);                                                            \
        }                                                                         \
    } while (0)

// ---------------------------------------------------------------------------
// CPU sequential merge sort (baseline)
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// GPU kernel: each thread merges one pair of sorted sub-arrays
// ---------------------------------------------------------------------------
__global__ void gpu_merge_kernel(int *src, int *dst, int n, int width)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int left = tid * 2 * width;

    if (left >= n) return;

    int mid   = min(left + width, n);
    int right = min(left + 2 * width, n);

    // Merge src[left..mid-1] and src[mid..right-1] into dst[left..right-1]
    int i = left, j = mid, k = left;
    while (i < mid && j < right)
        dst[k++] = (src[i] <= src[j]) ? src[i++] : src[j++];
    while (i < mid)   dst[k++] = src[i++];
    while (j < right) dst[k++] = src[j++];
}

// ---------------------------------------------------------------------------
// GPU merge sort: bottom-up, doubling the merge width each pass
// ---------------------------------------------------------------------------
void gpu_mergeSort(int **d_data, int **d_temp, int n)
{
    int threads = 256;

    for (int width = 1; width < n; width *= 2) {
        int num_merges = (n + 2 * width - 1) / (2 * width);
        int blocks = (num_merges + threads - 1) / threads;

        gpu_merge_kernel<<<blocks, threads>>>(*d_data, *d_temp, n, width);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Swap pointers for next pass
        int *tmp = *d_data;
        *d_data  = *d_temp;
        *d_temp  = tmp;
    }
}

// ---------------------------------------------------------------------------
// Verify two arrays are identical
// ---------------------------------------------------------------------------
int verify(int *a, int *b, int n)
{
    for (int i = 0; i < n; i++)
        if (a[i] != b[i]) return 0;
    return 1;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main()
{
    int N = 1 << 25; // 1M elements - same as your baseline run
    size_t bytes = N * sizeof(int);

    printf("=================================================\n");
    printf("  Merge Sort Benchmark: CPU vs GPU\n");
    printf("  N = %d elements\n", N);
    printf("=================================================\n\n");

    // Allocate and fill host arrays
    int *h_original = (int *)malloc(bytes);
    int *h_cpu      = (int *)malloc(bytes);
    int *h_gpu      = (int *)malloc(bytes);

    srand(42); // fixed seed for reproducibility
    for (int i = 0; i < N; i++)
        h_original[i] = rand() % 1000000;

    // -----------------------------------------------------------------------
    // CPU benchmark
    // -----------------------------------------------------------------------
    memcpy(h_cpu, h_original, bytes);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    cpu_mergeSort(h_cpu, 0, N - 1);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double cpu_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf("[CPU] Sequential merge sort:  %.4f seconds\n\n", cpu_time);

    // -----------------------------------------------------------------------
    // GPU benchmark
    // -----------------------------------------------------------------------
    int *d_data, *d_temp;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMalloc(&d_temp, bytes));

    CUDA_CHECK(cudaMemcpy(d_data, h_original, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_temp, h_original, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start));
    gpu_mergeSort(&d_data, &d_temp, N);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float gpu_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop));
    double gpu_time = gpu_ms / 1000.0;

    // Determine which buffer holds the sorted result
    // (after log2(N) pointer swaps, odd passes -> d_data, even -> check passes)
    int passes = 0;
    for (int w = 1; w < N; w *= 2) passes++;
    // After each pass we swap, so sorted data ends up in d_temp if passes is odd
    // (because we swap after writing to d_temp, making d_temp become d_data)
    CUDA_CHECK(cudaMemcpy(h_gpu, d_data, bytes, cudaMemcpyDeviceToHost));

    printf("[GPU] Parallel merge sort:    %.4f seconds  (%.2f ms)\n\n", gpu_time, gpu_ms);

    // -----------------------------------------------------------------------
    // Correctness check
    // -----------------------------------------------------------------------
    printf("Correctness check: %s\n\n",
           verify(h_cpu, h_gpu, N) ? "PASSED" : "FAILED - check pointer swap logic");

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    printf("=================================================\n");
    printf("  Speedup: %.2fx\n", cpu_time / gpu_time);
    printf("=================================================\n");

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
