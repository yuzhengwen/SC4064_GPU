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

__global__ void matrixMul(const float *A, const float *B, float *C,
                          int M, int K, int N)
{
    // Calculate which element of C this thread computes
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index (i)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index (j)

    // Boundary check
    if (row < M && col < N)
    {
        float sum = 0.0f;

        // C[row][col] = sum of A[row][k] * B[k][col] for all k
        for (int k = 0; k < K; k++)
        {
            // A is M×K, access element at (row, k)
            float a = A[row * K + k];
            // B is K×N, access element at (k, col)
            float b = B[k * N + col];
            sum += a * b;
        }

        C[row * N + col] = sum;
    }
}

void initMatrix(float *mat, int rows, int cols, float maxVal)
{
    for (int i = 0; i < rows * cols; i++)
    {
        mat[i] = (float)rand() / (float)RAND_MAX * maxVal;
    }
}

void initMatrixConstant(float *mat, int rows, int cols, float val)
{
    for (int i = 0; i < rows * cols; i++)
    {
        mat[i] = val;
    }
}

int main()
{
    // matrix dimensions
    const int M = 8192;
    const int K = 8192;
    const int N = 8192;

    const size_t bytesA = M * K * sizeof(float);
    const size_t bytesB = K * N * sizeof(float);
    const size_t bytesC = M * N * sizeof(float);

    printf("Matrix Multiplication: C = A × B\n");
    printf("=================================\n");
    printf("Matrix A: %d × %d (%.2f GB)\n", M, K, bytesA / 1e9);
    printf("Matrix B: %d × %d (%.2f GB)\n", K, N, bytesB / 1e9);
    printf("Matrix C: %d × %d (%.2f GB)\n", M, N, bytesC / 1e9);
    printf("\n");

    printf("Allocating host memory...\n");
    float *h_A = (float *)malloc(bytesA);
    float *h_B = (float *)malloc(bytesB);
    float *h_C = (float *)malloc(bytesC);

    if (!h_A || !h_B || !h_C)
    {
        fprintf(stderr, "Failed to allocate host memory!\n");
        return 1;
    }

    printf("Initializing matrices...\n");
    srand(12345);
    initMatrix(h_A, M, K, 1.0f); // Range [0, 1]
    initMatrix(h_B, K, N, 1.0f); // Range [0, 1]
    initMatrixConstant(h_C, M, N, 0.0f);

    printf("Allocating device memory...\n");
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));

    printf("Copying data to device...\n");
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, bytesC, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int blockSizes[][2] = {
        {8, 8},   // 64 threads per block
        {16, 16}, // 256 threads per block
        {32, 32}  // 1024 threads per block (max)
    };
    int numConfigs = sizeof(blockSizes) / sizeof(blockSizes[0]);

    printf("\nTesting different 2D block sizes:\n");
    printf("================================================\n");

    for (int config = 0; config < numConfigs; config++)
    {
        int blockX = blockSizes[config][0];
        int blockY = blockSizes[config][1];

        dim3 blockSize(blockX, blockY);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                      (M + blockSize.y - 1) / blockSize.y);

        printf("\nConfiguration %d:\n", config + 1);
        printf("----------------\n");
        printf("Block size: (%d, %d) = %d threads per block\n",
               blockX, blockY, blockX * blockY);
        printf("Grid size:  (%d, %d) = %d blocks\n",
               gridSize.x, gridSize.y, gridSize.x * gridSize.y);
        printf("Total threads: %d\n",
               gridSize.x * gridSize.y * blockX * blockY);
        printf("\nThread Index Calculation:\n");
        printf("  row = blockIdx.y * %d + threadIdx.y\n", blockY);
        printf("  col = blockIdx.x * %d + threadIdx.x\n", blockX);
        printf("  C[row][col] = Σ(A[row][k] * B[k][col]) for k=0 to %d\n", K - 1);
        printf("\n");

        // Warm-up run
        matrixMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed run
        CUDA_CHECK(cudaEventRecord(start));
        matrixMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        /* For learning:
         * FLOPS Calculation for Matrix Multiplication:
         *
         * For each element C[i][j]:
         * - K multiplications
         * - K additions (K-1 actually, but we count K for simplicity)
         * - Total: 2K operations per element
         *
         * Total operations: M × N × 2K
         * For 8192×8192×8192: 8192³ × 2 ≈ 1.1 trillion operations
         *
         * FLOPS = Operations / Time
         */
        double operations = 2.0 * M * N * K; // 2 ops (mul + add) per inner product element
        double seconds = milliseconds / 1000.0;
        double gflops = (operations / seconds) / 1e9;
        double tflops = gflops / 1000.0;

        printf("Performance:\n");
        printf("  Execution time: %.4f ms\n", milliseconds);
        printf("  Total operations: %.2e\n", operations);
        printf("  Performance: %.2f GFLOPS (%.4f TFLOPS)\n", gflops, tflops);
    }

    printf("Cleaning up...\n");
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
