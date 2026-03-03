#include "gpu_thrust.h"
#include "cuda_check.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

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
