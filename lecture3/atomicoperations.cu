#include <cuda_runtime.h>
#include <stdio.h>
#define N 1024 // number of data points
#define BIN 16 // number of histogram bins
__global__ void histogramKernel(int *data, int *hist, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        int bin = data[idx] % BIN; // map data to a bin
        atomicAdd(&hist[bin], 1);  // safely increment bin count
    }
}
int main()
{
    int h_data[N];         // input data
    int h_hist[BIN] = {0}; // histogram bins
    // Initialize input data
    for (int i = 0; i < N; i++)
        h_data[i] = i;
    int *d_data, *d_hist;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_hist, BIN * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_hist, h_hist, BIN * sizeof(int),
               cudaMemcpyHostToDevice);
    histogramKernel<<<(N + 255) / 256, 256>>>(d_data, d_hist, N);
    cudaMemcpy(h_hist, d_hist, BIN * sizeof(int),
               cudaMemcpyDeviceToHost);
    // Print histogram
    for (int i = 0; i < BIN; i++)
        printf("Bin %d: %d\n", i, h_hist[i]);
    cudaFree(d_data);
    cudaFree(d_hist);
    return 0;
}
