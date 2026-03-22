#include <stdio.h>
#include <cuda.h>
#define N 32 // array size (must be <= 32 for one warp)

// Inline: Compiler copies the function code into where it is called, instead of creating a function call
__inline__ __device__ float warpReduceSum(float val)
{
    unsigned int mask = 0xFFFFFFFF; // all threads active
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(mask, val, offset);
    return val;
}
__global__ void sumArrayWarp(float *A, float *result, int n)
{
    int tid = threadIdx.x;
    float val = (tid < n) ? A[tid] : 0.0f; // load or 0 if out of range
    // warp-level reduction
    val = warpReduceSum(val);
    // thread 0 of warp writes the final result
    if (tid == 0)
        *result = val;
}
int main()
{
    float h_A[N], h_result;
    for (int i = 0; i < N; i++)
        h_A[i] = i * 1.0f; // initialize with ones
    float *d_A, *d_result;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    sumArrayWarp<<<1, N>>>(d_A, d_result, N);
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Sum of array elements = %f\n", h_result);
    cudaFree(d_A);
    cudaFree(d_result);
    return 0;
}