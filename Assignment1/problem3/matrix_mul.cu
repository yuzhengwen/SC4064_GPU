/*
 * Problem 3: Matrix Multiplication
 * SC4064 GPU Programming Assignment 1
 * 
 * This program performs matrix multiplication C = A × B using CUDA
 * - Matrix dimensions: M=K=N=8192 (square matrices)
 * - Tests multiple 2D block sizes: 8×8, 16×16, 32×32
 * - Measures execution time and computes FLOPS
 * 
 * Algorithm:
 * - Each thread computes one element of C
 * - C[i][j] = sum(A[i][k] * B[k][j]) for k = 0 to K-1
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s (at line %s:%d)\n",cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(err); \
    } \
} while (0)

/*
 * CUDA Kernel: Matrix Multiplication (Naive Implementation)
 * 
 * Thread Index to Matrix Element Mapping:
 * 
 * 1. Calculate row index (i) for matrix C:
 *    i = blockIdx.y * blockDim.y + threadIdx.y
 *    - This tells us which row of C this thread is responsible for
 * 
 * 2. Calculate column index (j) for matrix C:
 *    j = blockIdx.x * blockDim.x + threadIdx.x
 *    - This tells us which column of C this thread is responsible for
 * 
 * 3. Compute the inner product:
 *    C[i][j] = Σ(A[i][k] * B[k][j]) for k=0 to K-1
 *    
 *    In linear memory (row-major):
 *    - A[i][k] is at position: A[i * K + k]
 *    - B[k][j] is at position: B[k * N + j]
 *    - C[i][j] is at position: C[i * N + j]
 * 
 * Example: Computing C[2][3] with K=4
 *    i=2, j=3
 *    C[2][3] = A[2][0]*B[0][3] + A[2][1]*B[1][3] + A[2][2]*B[2][3] + A[2][3]*B[3][3]
 * 
 * Memory Access Pattern:
 *    A: A[2*4+0], A[2*4+1], A[2*4+2], A[2*4+3]  (row 2, consecutive in memory)
 *    B: B[0*4+3], B[1*4+3], B[2*4+3], B[3*4+3]  (column 3, strided access)
 *    C: C[2*4+3]  (single write)
 * 
 * Performance Note:
 * - This is a naive O(N³) algorithm
 * - No shared memory optimization
 * - Not cache-optimized
 * - For better performance, use tiled/shared memory approaches
 */
__global__ void matrixMul(const float *A, const float *B, float *C,
                          int M, int K, int N) {
    // Calculate which element of C this thread computes
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Row index (i)
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Column index (j)
    
    // Boundary check
    if (row < M && col < N) {
        float sum = 0.0f;
        
        // Compute the inner product
        // C[row][col] = sum of A[row][k] * B[k][col] for all k
        for (int k = 0; k < K; k++) {
            // A is M×K, access element at (row, k)
            float a = A[row * K + k];
            
            // B is K×N, access element at (k, col)
            float b = B[k * N + col];
            
            // Accumulate the product
            sum += a * b;
        }
        
        // Store result in C (M×N matrix)
        C[row * N + col] = sum;
    }
}

// Initialize matrix with random values
void initMatrix(float *mat, int rows, int cols, float maxVal) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / (float)RAND_MAX * maxVal;
    }
}

// Initialize matrix with constant value
void initMatrixConstant(float *mat, int rows, int cols, float val) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = val;
    }
}

// CPU matrix multiplication for verification
void matrixMulCPU(const float *A, const float *B, float *C, 
                  int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Verify GPU results against CPU computation (check sample elements)
void verifyResult(const float *h_C_gpu, int M, int N, 
                  const float *A, const float *B, int K) {
    printf("Verifying results...\n");
    
    // Check a few random elements
    const int numChecks = 100;
    for (int check = 0; check < numChecks; check++) {
        int i = rand() % M;
        int j = rand() % N;
        
        // Compute expected value on CPU
        float expected = 0.0f;
        for (int k = 0; k < K; k++) {
            expected += A[i * K + k] * B[k * N + j];
        }
        
        float actual = h_C_gpu[i * N + j];
        float diff = fabs(expected - actual);
        float relError = diff / (fabs(expected) + 1e-5);
        
        if (relError > 1e-3) {  // Allow small floating-point errors
            printf("✗ Verification failed at C[%d][%d]: expected %f, got %f (rel error: %e)\n", 
                   i, j, expected, actual, relError);
            return;
        }
    }
    
    printf("✓ Verification passed (checked %d random elements)\n", numChecks);
}

int main() {
    // Matrix dimensions (square matrices)
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
    printf("Total memory: %.2f GB\n", (bytesA + bytesB + bytesC) / 1e9);
    printf("\n");
    
    // Allocate host memory
    printf("Allocating host memory...\n");
    float *h_A = (float*)malloc(bytesA);
    float *h_B = (float*)malloc(bytesB);
    float *h_C = (float*)malloc(bytesC);
    
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory!\n");
        return 1;
    }
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    srand(12345);
    initMatrix(h_A, M, K, 1.0f);  // Range [0, 1]
    initMatrix(h_B, K, N, 1.0f);  // Range [0, 1]
    initMatrixConstant(h_C, M, N, 0.0f);
    
    // Allocate device memory
    printf("Allocating device memory...\n");
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));
    
    // Copy input matrices to device
    printf("Copying data to device...\n");
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, bytesC, cudaMemcpyHostToDevice));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    /*
     * Test different block sizes
     * 
     * Trade-offs:
     * - Smaller blocks (8×8=64 threads): Lower occupancy, more blocks, less efficient
     * - Medium blocks (16×16=256 threads): Good balance, commonly used
     * - Larger blocks (32×32=1024 threads): Maximum threads per block, 
     *   but may limit occupancy due to register/shared memory constraints
     */
    int blockSizes[][2] = {
        {8, 8},    // 64 threads per block
        {16, 16},  // 256 threads per block
        {32, 32}   // 1024 threads per block (maximum for many GPUs)
    };
    int numConfigs = sizeof(blockSizes) / sizeof(blockSizes[0]);
    
    printf("\nTesting different 2D block sizes:\n");
    printf("================================================\n");
    
    for (int config = 0; config < numConfigs; config++) {
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
        printf("  C[row][col] = Σ(A[row][k] * B[k][col]) for k=0 to %d\n", K-1);
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
        
        /*
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
        double operations = 2.0 * M * N * K;  // 2 ops (mul + add) per inner product element
        double seconds = milliseconds / 1000.0;
        double gflops = (operations / seconds) / 1e9;
        double tflops = gflops / 1000.0;
        
        printf("Performance:\n");
        printf("  Execution time: %.4f ms\n", milliseconds);
        printf("  Total operations: %.2e\n", operations);
        printf("  Performance: %.2f GFLOPS (%.4f TFLOPS)\n", gflops, tflops);
        
        // Copy result back for first configuration (for verification)
        if (config == 0) {
            CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));
            verifyResult(h_C, M, N, h_A, h_B, K);
        }
    }
    
    printf("\n================================================\n");
    printf("Performance Analysis:\n");
    printf("================================================\n");
    printf("Expected observations:\n");
    printf("1. Larger block sizes generally perform better up to a point\n");
    printf("2. 16×16 often provides good balance between occupancy and efficiency\n");
    printf("3. 32×32 may not always be fastest due to resource constraints\n");
    printf("4. Performance depends on:\n");
    printf("   - Occupancy (warps per SM)\n");
    printf("   - Memory bandwidth utilization\n");
    printf("   - Register usage\n");
    printf("   - Shared memory usage (none in this naive implementation)\n");
    printf("\n");
    
    // Clean up
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
