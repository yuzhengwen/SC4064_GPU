#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                                                                      \
    do                                                                                                        \
    {                                                                                                         \
        cudaError_t err = call;                                                                               \
        if (err != cudaSuccess)                                                                               \
        {                                                                                                     \
            fprintf(stderr, "CUDA error: %s (at line %s:%d)\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(err);                                                                                        \
        }                                                                                                     \
    } while (0)

// Each thread computes one element: C[i] = A[i] + B[i]
// Global thread ID = blockIdx.x * blockDim.x + threadIdx.x => each thread id from 0 to N-1
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    // Calculate global 1d thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // If more threads than N size
    if (tid < N)
    {
        C[tid] = A[tid] + B[tid];
    }
}

void initVector(float *vec, int N)
{
    for (int i = 0; i < N; i++)
    {
        vec[i] = (float)rand() / (float)RAND_MAX * 100.0f;
    }
}

int main()
{
    const int N = 1 << 30; // 2^30
    const size_t bytes = N * sizeof(float);

    printf("Vector Addition\n");
    printf("Vector size: %d elements (%.2f GB per vector)\n", N, bytes / 1e9);
    printf("\n");

    printf("Allocating host memory...\n");
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    if (!h_A || !h_B || !h_C)
    {
        fprintf(stderr, "Failed to allocate host memory!\n");
        return 1;
    }

    printf("Initializing vectors with random values...\n");
    srand(100);
    initVector(h_A, N);
    initVector(h_B, N);

    printf("Allocating device memory...\n");
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    printf("Copying data to device...\n");
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    int blockSizes[] = {32, 64, 128, 256};
    int numTests = sizeof(blockSizes) / sizeof(blockSizes[0]);

    for (int i = 0; i < numTests; i++)
    {
        int blockSize = blockSizes[i];
        int gridSize = (N + blockSize - 1) / blockSize;

        printf("\nBlock size: %d threads\n", blockSize);
        printf("Grid size:  %d blocks\n", gridSize);
        printf("Total threads: %d\n", gridSize * blockSize);

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // Read that it's good to include this so that initialization overhead, or context-switching (from another job) does not affect timing
        vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaDeviceSynchronize()); // wait here for gpu

        // Timed run
        CUDA_CHECK(cudaEventRecord(start));
        vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop)); // wait for gpu

        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        double operations = (double)N; // no of operations equal to N (vector add)
        double seconds = milliseconds / 1000.0;
        double gflops = (operations / seconds) / 1e9;

        printf("Execution time: %.4f ms\n", milliseconds);
        printf("Performance:    %.2f GFLOPS\n", gflops);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    printf("\nCleaning up...\n");
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    printf("\nDone!\n");
    return 0;
}
