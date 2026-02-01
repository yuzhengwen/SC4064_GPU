#include <stdio.h>
__global__ void helloFromGPU() {
	int threadId = threadIdx.x; // Thread index within the block
	int blockId = blockIdx.x; // Block index within the grid
	int globalId = blockId * blockDim.x + threadId;
	printf("Hello World from thread %d (block %d, thread %d)\n",
	globalId, blockId, threadId);
}

int main() {
	// 4 blocks, 4 threads per block
	helloFromGPU<<<4, 4>>>();
	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();
	return 0;
}