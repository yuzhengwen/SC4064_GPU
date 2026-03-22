#include <stdio.h>
#include <cuda_runtime.h>
#define N 1024
#define BLOCK_SIZE 256
// --- Static shared memory kernel --
__global__ void vecAddStatic(float *A, float *B, float *C, int n)
{
    __shared__ float temp[BLOCK_SIZE]; // static shared memory
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < n)
    {
        temp[threadIdx.x] = A[tid];
        __syncthreads();
        C[tid] = temp[threadIdx.x] + B[tid];
    }
}
// --- Dynamic shared memory kernel --
__global__ void vecAddDynamic(float *A, float *B, float *C, int n)
{
    extern __shared__ float temp[]; // dynamic shared memory
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < n)
    {
        temp[threadIdx.x] = A[tid];
        __syncthreads();
        C[tid] = temp[threadIdx.x] + B[tid];
    }
}
int main()
{
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    size_t size = N * sizeof(float);
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size); // Initialize vectors
    for (int i = 0; i < N; i++)
    {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // --- Run static shared memory kernel --
    vecAddStatic<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    printf("Static shared memory result: C[0]=%f, C[1]=%f, C[N-1]=%f\n",
           h_C[0], h_C[1], h_C[N - 1]);
    // --- Run dynamic shared memory kernel --
    vecAddDynamic<<<gridSize, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    printf("Dynamic shared memory result: C[0]=%f, C[1]=%f, C[N-1]=%f\n",
           h_C[0], h_C[1], h_C[N - 1]);
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
