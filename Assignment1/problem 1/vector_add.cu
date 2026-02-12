/*
 * Problem 1: Vector Addition
 * SC4064 GPU Programming Assignment 1
 * 
 * This program performs vector addition C = A + B using CUDA
 * - Vectors of length 2^30 (1,073,741,824 elements)
 * - Tests block sizes: 32, 64, 128, 256
 * - Measures execution time and computes FLOPS
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Error checking macro from Week 2 lecture
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
 * CUDA Kernel: Vector Addition
 * 
 * Each thread computes one element: C[i] = A[i] + B[i]
 * 
 * Thread Index Calculation:
 * - Global thread ID = blockIdx.x * blockDim.x + threadIdx.x
 * - This gives each thread a unique index from 0 to N-1
 * 
 * Boundary Check:
 * - We check (tid < N) because the total number of threads may exceed N
 * - Example: N=1000, blockSize=256, gridSize=4 → 1024 threads total
 *   Threads 1000-1023 should do nothing
 */
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    // Calculate global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check - only process if within vector bounds
    if (tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}

/*
 * Initialize vector with random values in range [0.0, 100.0]
 */
void initVector(float *vec, int N) {
    for (int i = 0; i < N; i++) {
        vec[i] = (float)rand() / (float)RAND_MAX * 100.0f;
    }
}

/*
 * Verify results by checking a few elements
 * In production, you'd check all elements, but for N=2^30, 
 * checking a sample is more practical
 */
void verifyResult(const float *A, const float *B, const float *C, int N) {
    const int numChecks = 1000;
    for (int i = 0; i < numChecks; i++) {
        int idx = rand() % N;
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
    // Vector size: 2^30 elements
    const int N = 1 << 30;  // 1,073,741,824
    const size_t bytes = N * sizeof(float);
    
    printf("Vector Addition\n");
    printf("===============\n");
    printf("Vector size: %d elements (%.2f GB per vector)\n", N, bytes / 1e9);
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
    
    // Initialize vectors
    printf("Initializing vectors with random values...\n");
    srand(12345);  // Fixed seed for reproducibility
    initVector(h_A, N);
    initVector(h_B, N);
    
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
    
    // Test different block sizes
    int blockSizes[] = {32, 64, 128, 256};
    int numTests = sizeof(blockSizes) / sizeof(blockSizes[0]);
    
    printf("\nTesting different block sizes:\n");
    printf("================================================\n");
    
    for (int i = 0; i < numTests; i++) {
        int blockSize = blockSizes[i];
        
        // Calculate grid size (number of blocks needed)
        // We use ceiling division: (N + blockSize - 1) / blockSize
        // This ensures we have enough threads to cover all N elements
        int gridSize = (N + blockSize - 1) / blockSize;
        
        printf("\nBlock size: %d threads\n", blockSize);
        printf("Grid size:  %d blocks\n", gridSize);
        printf("Total threads: %d\n", gridSize * blockSize);
        
        // Create CUDA events for timing
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        // Warm-up run (not timed)
        vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Timed run
        CUDA_CHECK(cudaEventRecord(start));
        vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        // Calculate execution time
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        
        // Calculate FLOPS
        // Operations: N additions
        // FLOPS = Operations / Time (in seconds)
        double operations = (double)N;
        double seconds = milliseconds / 1000.0;
        double gflops = (operations / seconds) / 1e9;
        
        printf("Execution time: %.4f ms\n", milliseconds);
        printf("Performance:    %.2f GFLOPS\n", gflops);
        
        // Clean up events
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    // Copy result back to host for verification
    printf("\nCopying result back to host...\n");
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    
    // Verify result
    printf("Verifying result...\n");
    verifyResult(h_A, h_B, h_C, N);
    
    // Clean up
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
