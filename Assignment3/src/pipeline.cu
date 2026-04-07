#include "pipeline.cuh"
#include <cmath>
#include <cstdio>

// ─────────────────────────────────────────────────────────────────────────────
// Gaussian kernel weights (normalised by 273).
// Stored as a constant array in GPU constant memory for fast broadcast reads.
// ─────────────────────────────────────────────────────────────────────────────
__constant__ float c_gauss[5][5] = {
    { 1.f/273,  4.f/273,  7.f/273,  4.f/273,  1.f/273 },
    { 4.f/273, 16.f/273, 26.f/273, 16.f/273,  4.f/273 },
    { 7.f/273, 26.f/273, 41.f/273, 26.f/273,  7.f/273 },
    { 4.f/273, 16.f/273, 26.f/273, 16.f/273,  4.f/273 },
    { 1.f/273,  4.f/273,  7.f/273,  4.f/273,  1.f/273 },
};


// ═════════════════════════════════════════════════════════════════════════════
// STAGE 1 — Gaussian Blur (shared memory tiling with halo cells)
// ═════════════════════════════════════════════════════════════════════════════
__global__ void gaussianBlurKernel(
    const uint8_t* __restrict__ in,
    uint8_t*       __restrict__ out,
    int width, int height)
{
    // Shared memory tile dimensions (centre + halo on each side).
    const int SMEM_W = TILE_W + 2 * GAUSS_RADIUS;
    const int SMEM_H = TILE_H + 2 * GAUSS_RADIUS;

    // [TASK 1] Shared memory array large enough for output tile + halo.
    __shared__ uint8_t smem[SMEM_H][SMEM_W];

    // [TASK 2] Global (x, y) output pixel this thread is responsible for.
    int out_x = blockIdx.x * TILE_W + threadIdx.x;
    int out_y = blockIdx.y * TILE_H + threadIdx.y;

    // [TASK 3] Cooperatively load the full SMEM_H x SMEM_W input tile.
    //
    // Top-left corner of the shared tile in global coordinates.
    int tile_start_x = blockIdx.x * TILE_W - GAUSS_RADIUS;
    int tile_start_y = blockIdx.y * TILE_H - GAUSS_RADIUS;

    // Strided loop: each thread loads one or more pixels using its
    // linearised ID within the block.
    int tid        = threadIdx.y * TILE_W + threadIdx.x;
    int block_size = TILE_W * TILE_H;
    int smem_size  = SMEM_W * SMEM_H;

    for (int idx = tid; idx < smem_size; idx += block_size) {
        int sm_row = idx / SMEM_W;
        int sm_col = idx % SMEM_W;

        // Clamp global coords to [0, width-1] / [0, height-1] (border replication).
        int gx = min(max(tile_start_x + sm_col, 0), width  - 1);
        int gy = min(max(tile_start_y + sm_row, 0), height - 1);

        smem[sm_row][sm_col] = in[gy * width + gx];
    }

    // [TASK 4] Barrier — all data loaded before any thread reads smem.
    __syncthreads();

    // [TASK 5 & 6] 5x5 convolution; only write for in-bounds output pixels.
    if (out_x < width && out_y < height) {
        float sum = 0.f;
        // threadIdx offsets are already at the centre of the halo region.
        for (int ki = 0; ki < 5; ki++) {
            for (int kj = 0; kj < 5; kj++) {
                sum += c_gauss[ki][kj] * smem[threadIdx.y + ki][threadIdx.x + kj];
            }
        }
        out[out_y * width + out_x] = (uint8_t)min(max((int)roundf(sum), 0), 255);
    }
}


// ═════════════════════════════════════════════════════════════════════════════
// STAGE 2 — Sobel Edge Detection
// ═════════════════════════════════════════════════════════════════════════════
//
//   Gx = [[-1, 0, 1],     Gy = [[ 1,  2,  1],
//         [-2, 0, 2],           [ 0,  0,  0],
//         [-1, 0, 1]]           [-1, -2, -1]]
//
__global__ void sobelKernel(
    const uint8_t* __restrict__ in,
    uint8_t*       __restrict__ out,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Helper lambda — clamp-to-edge fetch.
    // (Can't use a true lambda in device code on older toolchains, so use a macro.)
#define FETCH(dx, dy) \
    ((float)in[ min(max(y+(dy), 0), height-1) * width + min(max(x+(dx), 0), width-1) ])

    float gx = -1*FETCH(-1,-1) + 0*FETCH(0,-1) + 1*FETCH(1,-1)
               -2*FETCH(-1, 0) + 0*FETCH(0, 0) + 2*FETCH(1, 0)
               -1*FETCH(-1, 1) + 0*FETCH(0, 1) + 1*FETCH(1, 1);

    float gy =  1*FETCH(-1,-1) + 2*FETCH(0,-1) + 1*FETCH(1,-1)
               +0*FETCH(-1, 0) + 0*FETCH(0, 0) + 0*FETCH(1, 0)
               -1*FETCH(-1, 1) - 2*FETCH(0, 1) - 1*FETCH(1, 1);

#undef FETCH

    float mag = sqrtf(gx*gx + gy*gy);
    out[y * width + x] = (uint8_t)min((int)mag, 255);
}


// ═════════════════════════════════════════════════════════════════════════════
// STAGE 3A — Histogram Kernel
// ═════════════════════════════════════════════════════════════════════════════
//
// Uses a per-block shared histogram to reduce contention on the 256 global
// counters (optional optimisation mentioned in the spec).
//
__global__ void histogramKernel(
    const uint8_t*  __restrict__ in,
    unsigned int*   hist,
    int width, int height)
{
    // Per-block shared histogram — 256 bins.
    __shared__ unsigned int s_hist[256];

    // Zero-initialise the shared histogram cooperatively.
    for (int b = threadIdx.x + threadIdx.y * blockDim.x; b < 256;
         b += blockDim.x * blockDim.y) {
        s_hist[b] = 0u;
    }
    __syncthreads();

    // Each thread processes one pixel.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        uint8_t val = in[y * width + x];
        atomicAdd(&s_hist[val], 1u);
    }
    __syncthreads();

    // Flush block-local counts to global memory.
    for (int b = threadIdx.x + threadIdx.y * blockDim.x; b < 256;
         b += blockDim.x * blockDim.y) {
        if (s_hist[b] > 0u)
            atomicAdd(&hist[b], s_hist[b]);
    }
}


// ═════════════════════════════════════════════════════════════════════════════
// STAGE 3C — Equalisation Kernel
// ═════════════════════════════════════════════════════════════════════════════
//
// new_val = round( (CDF[old_val] - cdf_min) / (W*H - cdf_min) * 255 )
//
// `cdf` comes from thrust::exclusive_scan over the histogram, so:
//   cdf[0] = 0, cdf[i] = number of pixels with intensity < i.
//
__global__ void equalizeKernel(
    const uint8_t* __restrict__ in,
    uint8_t*       __restrict__ out,
    const float*   cdf,
    float          cdf_min,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    uint8_t old_val  = in[y * width + x];
    float   total    = (float)(width * height);
    float   new_valf = roundf((cdf[old_val] - cdf_min) / (total - cdf_min) * 255.f);
    out[y * width + x] = (uint8_t)min(max((int)new_valf, 0), 255);
}
