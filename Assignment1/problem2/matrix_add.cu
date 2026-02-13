#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

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

__global__ void matrixAdd1D(const float *A, const float *B, float *C,
                            int rows, int cols)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int i = tid / cols; // Row index
    int j = tid % cols; // Column index

    // Boundary check
    if (i < rows && j < cols)
    {
        int index = i * cols + j;
        C[index] = A[index] + B[index];
    }
}

__global__ void matrixAdd2D(const float *A, const float *B, float *C,
                            int rows, int cols)
{
    // Calculate 2D thread indices directly
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column

    // Boundary check
    if (i < rows && j < cols)
    {
        // Convert to linear index and perform addition
        int index = i * cols + j;
        C[index] = A[index] + B[index];
    }
}

// fill random values
void initMatrix(float *mat, int rows, int cols, float maxVal)
{
    for (int i = 0; i < rows * cols; i++)
    {
        mat[i] = (float)rand() / (float)RAND_MAX * maxVal;
    }
}

int main()
{
    const int rows = 8192;
    const int cols = 8192;
    const int N = rows * cols;
    const size_t bytes = N * sizeof(float);

    printf("Matrix Addition\n");
    printf("Matrix size: %d x %d (%d elements)\n", rows, cols, N);
    printf("Memory per matrix: %.2f GB\n", bytes / 1e9);
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

    // Initialize matrices -> A, B random, C all 0
    printf("Initializing matrices...\n");
    srand(12345);
    initMatrix(h_A, rows, cols, 100.0f);
    initMatrix(h_B, rows, cols, 100.0f);
    for (int i = 0; i < rows * cols; i++)
    {
        h_C[i] = 0.0f;
    }

    printf("Allocating device memory...\n");
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    printf("Copying data to device...\n");
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    printf("\n================================================\n");
    printf("Configuration 1: 1D Grid with 1D Blocks\n");
    printf("================================================\n\n");

    {
        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;

        printf("Configuration:\n");
        printf("  Block size: %d threads\n", blockSize);
        printf("  Grid size:  %d blocks\n", gridSize);
        printf("  Total threads: %d\n\n", gridSize * blockSize);

        // Warm-up run (same as before)
        matrixAdd1D<<<gridSize, blockSize>>>(d_A, d_B, d_C, rows, cols);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Proper run
        CUDA_CHECK(cudaEventRecord(start));
        matrixAdd1D<<<gridSize, blockSize>>>(d_A, d_B, d_C, rows, cols);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        double operations = (double)N;
        double seconds = milliseconds / 1000.0;
        double gflops = (operations / seconds) / 1e9;

        printf("Performance:\n");
        printf("  Execution time: %.4f ms\n", milliseconds);
        printf("  GFLOPS:         %.2f\n", gflops);
    }

    printf("\n================================================\n");
    printf("Configuration 2: 2D Grid with 2D Blocks\n");
    printf("================================================\n\n");

    {
        dim3 blockSize(16, 16); // 16x16 = 256 threads per block
        dim3 gridSize((cols + blockSize.x - 1) / blockSize.x,
                      (rows + blockSize.y - 1) / blockSize.y);

        printf("  Block size: (%d, %d) = %d threads\n",
               blockSize.x, blockSize.y, blockSize.x * blockSize.y);
        printf("  Grid size:  (%d, %d) = %d blocks\n",
               gridSize.x, gridSize.y, gridSize.x * gridSize.y);
        printf("  Total threads: %d\n\n",
               gridSize.x * gridSize.y * blockSize.x * blockSize.y);

        // Warm-up run
        matrixAdd2D<<<gridSize, blockSize>>>(d_A, d_B, d_C, rows, cols);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed run
        CUDA_CHECK(cudaEventRecord(start));
        matrixAdd2D<<<gridSize, blockSize>>>(d_A, d_B, d_C, rows, cols);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        double operations = (double)N;
        double seconds = milliseconds / 1000.0;
        double gflops = (operations / seconds) / 1e9;

        printf("Performance:\n");
        printf("  Execution time: %.4f ms\n", milliseconds);
        printf("  GFLOPS:         %.2f\n", gflops);
    }

    // Clean up
    printf("\nCleaning up...\n");
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    printf("\nDone!\n");
    return 0;
}
