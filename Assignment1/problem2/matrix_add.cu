/*
 * Problem 2: Matrix Addition
 * SC4064 GPU Programming Assignment 1
 * 
 * This program performs matrix addition C = A + B using CUDA
 * - Matrix size: 8192 x 8192
 * - Implements two configurations:
 *   1) 1D grid with 1D blocks
 *   2) 2D grid with 2D blocks
 * - Compares performance between the two approaches
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// Error checking macro
#define CUDA_CHECK(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

/*
 * CUDA Kernel: Matrix Addition (1D Configuration)
 * 
 * Thread Index Calculation:
 * 1. Calculate global linear thread ID:
 *    tid = blockIdx.x * blockDim.x + threadIdx.x
 * 
 * 2. Map linear ID to 2D matrix indices:
 *    row (i) = tid / cols    (integer division gives row number)
 *    col (j) = tid % cols    (remainder gives column number)
 * 
 * Example: For a 4x4 matrix (cols=4)
 *    tid=0  → i=0/4=0, j=0%4=0 → (0,0)
 *    tid=1  → i=1/4=0, j=1%4=1 → (0,1)
 *    tid=4  → i=4/4=1, j=4%4=0 → (1,0)
 *    tid=5  → i=5/4=1, j=5%4=1 → (1,1)
 * 
 * 3. Convert 2D indices to linear memory address:
 *    index = i * cols + j
 *    (Row-major ordering: rows are stored contiguously in memory)
 */
__global__ void matrixAdd1D(const float *A, const float *B, float *C, 
                            int rows, int cols) {
    // Step 1: Calculate global linear thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Step 2: Map to 2D matrix indices
    int i = tid / cols;  // Row index
    int j = tid % cols;  // Column index
    
    // Step 3: Boundary check
    if (i < rows && j < cols) {
        // Step 4: Convert to linear index and perform addition
        int index = i * cols + j;
        C[index] = A[index] + B[index];
    }
}

/*
 * CUDA Kernel: Matrix Addition (2D Configuration)
 * 
 * Thread Index Calculation (More Intuitive for 2D Data):
 * 1. Calculate row index (i):
 *    i = blockIdx.y * blockDim.y + threadIdx.y
 *    - blockIdx.y: which block row we're in
 *    - blockDim.y: threads per block in y-dimension
 *    - threadIdx.y: thread position within block
 * 
 * 2. Calculate column index (j):
 *    j = blockIdx.x * blockDim.x + threadIdx.x
 *    - blockIdx.x: which block column we're in
 *    - blockDim.x: threads per block in x-dimension
 *    - threadIdx.x: thread position within block
 * 
 * Example: blockDim = (16, 16), blockIdx = (2, 3), threadIdx = (5, 7)
 *    i = 3 * 16 + 7 = 55 (row 55)
 *    j = 2 * 16 + 5 = 37 (column 37)
 *    → Access element at position (55, 37)
 * 
 * 3. Convert to linear index:
 *    index = i * cols + j
 */
__global__ void matrixAdd2D(const float *A, const float *B, float *C, 
                            int rows, int cols) {
    // Calculate 2D thread indices directly
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // Row
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // Column
    
    // Boundary check
    if (i < rows && j < cols) {
        // Convert to linear index and perform addition
        int index = i * cols + j;
        C[index] = A[index] + B[index];
    }
}

// Initialize matrix with random values
void initMatrix(float *mat, int rows, int cols, float maxVal) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / (float)RAND_MAX * maxVal;
    }
}

// Initialize matrix with a constant value
void initMatrixConstant(float *mat, int rows, int cols, float val) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = val;
    }
}

// Verify results
void verifyResult(const float *A, const float *B, const float *C, 
                  int rows, int cols) {
    const int numChecks = 1000;
    for (int i = 0; i < numChecks; i++) {
        int idx = rand() % (rows * cols);
        float expected = A[idx] + B[idx];
        if (fabs(C[idx] - expected) > 1e-5) {
            fprintf(stderr, "Verification failed at index %d: expected %f, got %f\n", 
                    idx, expected, C[idx]);
            return;
        }
    }
    printf("✓ Verification passed (checked %d random elements)\n", numChecks);
}

int main() {
    // Matrix dimensions
    const int rows = 8192;
    const int cols = 8192;
    const int N = rows * cols;
    const size_t bytes = N * sizeof(float);
    
    printf("Matrix Addition\n");
    printf("===============\n");
    printf("Matrix size: %d x %d (%d elements)\n", rows, cols, N);
    printf("Memory per matrix: %.2f GB\n", bytes / 1e9);
    printf("\n");
    
    // Allocate host memory
    printf("Allocating host memory...\n");
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory!\n");
        return 1;
    }
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    srand(12345);
    initMatrix(h_A, rows, cols, 100.0f);
    initMatrix(h_B, rows, cols, 100.0f);
    initMatrixConstant(h_C, rows, cols, 0.0f);
    
    // Allocate device memory
    printf("Allocating device memory...\n");
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    
    // Copy data to device
    printf("Copying data to device...\n");
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    printf("\n================================================\n");
    printf("Configuration 1: 1D Grid with 1D Blocks\n");
    printf("================================================\n\n");
    
    /*
     * 1D Configuration
     * - Total threads needed: rows * cols = 67,108,864
     * - Block size: 256 threads (good balance)
     * - Grid size: (N + blockSize - 1) / blockSize
     */
    {
        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;
        
        printf("Thread Index Calculation:\n");
        printf("  tid = blockIdx.x * blockDim.x + threadIdx.x\n");
        printf("  i (row) = tid / %d\n", cols);
        printf("  j (col) = tid %% %d\n", cols);
        printf("  index = i * %d + j\n\n", cols);
        
        printf("Configuration:\n");
        printf("  Block size: %d threads\n", blockSize);
        printf("  Grid size:  %d blocks\n", gridSize);
        printf("  Total threads: %d\n\n", gridSize * blockSize);
        
        // Warm-up run
        matrixAdd1D<<<gridSize, blockSize>>>(d_A, d_B, d_C, rows, cols);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Timed run
        CUDA_CHECK(cudaEventRecord(start));
        matrixAdd1D<<<gridSize, blockSize>>>(d_A, d_B, d_C, rows, cols);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        
        // Calculate FLOPS (1 addition per element)
        double operations = (double)N;
        double seconds = milliseconds / 1000.0;
        double gflops = (operations / seconds) / 1e9;
        
        printf("Performance:\n");
        printf("  Execution time: %.4f ms\n", milliseconds);
        printf("  GFLOPS:         %.2f\n", gflops);
        
        // Verify
        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
        verifyResult(h_A, h_B, h_C, rows, cols);
    }
    
    printf("\n================================================\n");
    printf("Configuration 2: 2D Grid with 2D Blocks\n");
    printf("================================================\n\n");
    
    /*
     * 2D Configuration
     * - Block dimensions: (16, 16) = 256 threads per block
     * - Grid dimensions: Calculate based on matrix size
     *   gridDim.x = (cols + blockDim.x - 1) / blockDim.x
     *   gridDim.y = (rows + blockDim.y - 1) / blockDim.y
     * 
     * Benefits:
     * - More intuitive mapping to 2D data
     * - Better spatial locality
     * - Natural alignment with matrix structure
     */
    {
        dim3 blockSize(16, 16);  // 16x16 = 256 threads per block
        dim3 gridSize((cols + blockSize.x - 1) / blockSize.x,
                      (rows + blockSize.y - 1) / blockSize.y);
        
        printf("Thread Index Calculation:\n");
        printf("  i (row) = blockIdx.y * blockDim.y + threadIdx.y\n");
        printf("  j (col) = blockIdx.x * blockDim.x + threadIdx.x\n");
        printf("  index = i * %d + j\n\n", cols);
        
        printf("Configuration:\n");
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
        
        // Calculate FLOPS
        double operations = (double)N;
        double seconds = milliseconds / 1000.0;
        double gflops = (operations / seconds) / 1e9;
        
        printf("Performance:\n");
        printf("  Execution time: %.4f ms\n", milliseconds);
        printf("  GFLOPS:         %.2f\n", gflops);
        
        // Verify
        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
        verifyResult(h_A, h_B, h_C, rows, cols);
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
