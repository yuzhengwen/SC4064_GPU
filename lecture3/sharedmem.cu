#include <cuda_runtime.h>
#include <stdio.h>
#define BLOCK_SIZE 2048
__global__ void sharedHeavyKernel(float *out, const float *in, int N)
{
    extern __shared__ float sdata[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ltid = threadIdx.x;
    if (tid < N)
    {
        // Load to shared memory
        sdata[ltid] = in[tid];
        __syncthreads();
        // Reuse shared memory multiple times
        float val = sdata[ltid];
#pragma unroll 32
        for (int i = 0; i < 32; i++)
        {
            val = val * 1.0001f + sdata[(ltid + i) % blockDim.x];
        }
        out[tid] = val;
    }
}
void runTest(cudaFuncCache cacheConfig, const char *label)
{
    const int N = 1 << 24;
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaFuncSetCacheConfig(sharedHeavyKernel, cacheConfig);
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + block.x - 1) / block.x);
    size_t shmem = BLOCK_SIZE * sizeof(float);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 500; i++)
    {
        sharedHeavyKernel<<<grid, block, shmem>>>(d_out, d_in, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("%s: %.3f ms\n", label, ms);
    cudaFree(d_in);
    cudaFree(d_out);
}
int main()
{
    printf("No preference:\n");
    runTest(cudaFuncCachePreferNone, "  None");
    printf("Prefer shared memory:\n");
    runTest(cudaFuncCachePreferShared, "  PreferShared");
    printf("Prefer L1 cache:\n");
    runTest(cudaFuncCachePreferL1, "  PreferL1");
    printf("Equal:\n");
    runTest(cudaFuncCachePreferEqual, "  Equal");
    return 0;
}
