#include "gpu_thrust.h"
#include "cuda_check.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

float run_thrust_sort(const int *h_src, int *h_dst, int n)
{
    size_t bytes = (size_t)n * sizeof(int);

    int *d_arr;
    CUDA_CHECK(cudaMalloc(&d_arr, bytes));
    CUDA_CHECK(cudaMemcpy(d_arr, h_src, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start));
    thrust::sort(thrust::device_ptr<int>(d_arr),
                 thrust::device_ptr<int>(d_arr + n));
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));

    CUDA_CHECK(cudaMemcpy(h_dst, d_arr, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_arr));
    return ms;
}
