#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

#include "cuda_check.h"
#include "cpu_sort.h"
#include "gpu_naive.h"
#include "gpu_smem.h"
#include "gpu_thrust.h"

static int verify(int *ref, int *result, int n)
{
    for (int i = 0; i < n; i++)
        if (ref[i] != result[i]) return 0;
    return 1;
}

static void print_result(const char *label, double cpu_ms, double this_ms, int correct)
{
    printf("  %-34s %8.2f ms   speedup: %6.2fx   %s\n",
           label, this_ms, cpu_ms / this_ms,
           correct ? "PASSED" : "FAILED");
}

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
    double cpu_ms = ((t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9) * 1000.0;
    printf("[CPU] Sequential merge sort: %.2f ms\n", cpu_ms);

    printf("\n%-36s %10s   %14s   %s\n", "Variant", "Time (ms)", "vs CPU", "Correct?");
    printf("-----------------------------------------------------------------\n");

    // Shared device buffers reused across GPU variants
    int *d_data, *d_temp;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMalloc(&d_temp, bytes));
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // Naive GPU (prints per-pass table internally)
    printf("\n[Naive GPU - per-pass breakdown]\n");
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
