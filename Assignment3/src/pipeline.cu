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
//
// Background:
//   Each output pixel is a weighted average of its 5x5 neighbourhood.
//   Neighbouring output pixels share input pixels, so loading the input tile
//   into shared memory reduces global memory traffic significantly.
//   The shared tile must be larger than the output tile by GAUSS_RADIUS pixels
//   on every side — these extra pixels are called "halo cells".
//
// Shared memory layout:
//
//   ┌────────────────────────────┐  ← (TILE_W + 2*GAUSS_RADIUS) wide
//   │  halo  │  halo   │  halo   │  } GAUSS_RADIUS rows of halo
//   ├────────┼─────────┼─────────┤
//   │  halo  │ OUTPUT  │  halo   │  } TILE_H rows of output pixels
//   ├────────┼─────────┼─────────┤
//   │  halo  │  halo   │  halo   │  } GAUSS_RADIUS rows of halo
//   └────────────────────────────┘
//
// Your tasks:
//   1. Declare shared memory with the correct halo-extended dimensions.
//   2. Map each thread to a global (x, y) position.
//   3. Load the centre pixels AND halo pixels into shared memory cooperatively.
//      (Some threads may need to load more than one pixel.)
//   4. __syncthreads() before any computation.
//   5. Apply the 5x5 convolution from shared memory for in-bounds threads.
//   6. Write the result to `out`.
//
// ─────────────────────────────────────────────────────────────────────────────
__global__ void gaussianBlurKernel(
    const uint8_t* __restrict__ in,
    uint8_t*       __restrict__ out,
    int width, int height)
{
        // Shared memory tile dimensions (centre + halo on each side).
    const int SMEM_W = TILE_W + 2 * GAUSS_RADIUS; 
    const int SMEM_H = TILE_H + 2 * GAUSS_RADIUS; 

    // TODO: Declare shared memory array of size SMEM_H x SMEM_W.
	__shared__ uint8_t smem[SMEM_H][SMEM_W];

     // TODO: Compute the global (x, y) output pixel this thread is
    //            responsible for.
    int out_x = blockIdx.x * TILE_W + threadIdx.x;
    int out_y = blockIdx.y * TILE_H + threadIdx.y;

    // TODO: Load shared memory cooperatively.
    //
    //   The input tile starts at global coordinates:
    //     tile_start_x = blockIdx.x * TILE_W - GAUSS_RADIUS
    //     tile_start_y = blockIdx.y * TILE_H - GAUSS_RADIUS
    //
    //   Each thread should load at least its own pixel at (threadIdx.x + GAUSS_RADIUS,
    //   threadIdx.y + GAUSS_RADIUS) in shared memory, plus any halo pixels it is
    //   responsible for. common strategy: iterate over the SMEM_H x SMEM_W region
    //   using a strided loop over the linearised thread index.
    //
    //   Boundary condition: clamp out-of-bounds global coordinates to [0, width-1]
    //   and [0, height-1] before indexing into `in`.

    int tile_start_x = blockIdx.x * TILE_W - GAUSS_RADIUS;
    int tile_start_y = blockIdx.y * TILE_H - GAUSS_RADIUS;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y; 
    int total_pixels  = SMEM_W * SMEM_H;          

    for (int idx = tid; idx < total_pixels; idx += total_threads) {
        int sm_y = idx / SMEM_W;
        int sm_x = idx % SMEM_W;

        int gx = max(0, min(tile_start_x + sm_x, width - 1));
        int gy = max(0, min(tile_start_y + sm_y, height - 1));

		smem[sm_y][sm_x] = in[gy * width + gx];
    }

    // TODO: Synchronise all threads before computing the convolution.

    __syncthreads(); 

    // TODO: Apply the 5x5 Gaussian convolution from shared memory.
    //
    //   Each thread computes one output pixel.
    //     out_x = blockIdx.x * TILE_W + threadIdx.x;
    //     out_y = blockIdx.y * TILE_H + threadIdx.y;
    //   Only threads whose (out_x, out_y) is within [0, width) x [0, height)
    //   should write to `out`.
    //
    //   Sum over ki = 0..4, kj = 0..4:
    //     sum += c_gauss[ki][kj] * smem[threadIdx.y + ki][threadIdx.x + kj]
    //
    //   Clamp (use roundf) the result to [0, 255] and cast to uint8_t before storing
    //   i.e., (uint8_t)min(max((int)roundf(sum), 0), 255);
    
    if (out_x < width && out_y < height) {
        float sum = 0.f;
        for (int ki = 0; ki < 5; ki++) {
            for (int kj = 0; kj < 5; kj++) {
                //sum += c_gauss[ki][kj] * smem[threadIdx.y + ki][threadIdx.x + kj];
				sum += c_gauss[ki][kj] * (float)smem[threadIdx.y + ki][threadIdx.x + kj];
            }
        }
        // FIX 3: Use roundf to match the reference implementation exactly
        out[out_y * width + out_x] = (uint8_t)max(0, min((int)roundf(sum), 255));
    }
}


// ═════════════════════════════════════════════════════════════════════════════
// STAGE 2 — Sobel Edge Detection
// ═════════════════════════════════════════════════════════════════════════════
//
// Background
//   Two 3x3 kernels (Gx, Gy) measure intensity gradient in x and y directions.
//   Gradient magnitude = sqrt(Gx^2 + Gy^2), clamped to [0, 255].
//
//   Gx = [[-1, 0, 1],     Gy = [[ 1,  2,  1],
//         [-2, 0, 2],           [ 0,  0,  0],
//         [-1, 0, 1]]           [-1, -2, -1]]
//
// Both Gx and Gy must be computed in this single kernel.
// Shared memory tiling is optional but encouraged.
// Use clamp-to-edge for boundary pixels.
//
// ─────────────────────────────────────────────────────────────────────────────
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
    //out[y * width + x] = (uint8_t)min(max((int)roundf(mag), 0), 255);
	out[y * width + x] = (uint8_t)min((int)mag, 255);
}


// ═════════════════════════════════════════════════════════════════════════════
// STAGE 3A — Histogram Kernel
// ═════════════════════════════════════════════════════════════════════════════
//
// Background:
//   Count how many pixels have each intensity value (0–255).
//   Many threads will try to increment the same bin simultaneously,
//   so atomic operations are required.
//
// `hist` is a device array of 256 unsigned ints, zero-initialised before launch.
//
// Optimisation hint (optional, but worth attempting):
//   Use a per-block shared memory histogram (256 unsigned ints), accumulate
//   locally with __atomicAdd on shared memory, then flush to global memory
//   once per block. This reduces contention on the 256 global counters.
//
// ─────────────────────────────────────────────────────────────────────────────
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
// STAGE 3B — CDF on host (solution given in multigpu.cu)
// ═════════════════════════════════════════════════════════════════════════════

// ═════════════════════════════════════════════════════════════════════════════
// STAGE 3C — Equalisation Kernel
// ═════════════════════════════════════════════════════════════════════════════
//
// Background:
//   Remap each pixel using:
//     new_val = round((CDF[old_val] - cdf_min) / (W*H - cdf_min) * 255)
//
// `cdf` is a device array of 256 floats from thrust::exclusive_scan, so:
//  cdf[i] = number of pixels with intensity STRICTLY LESS THAN i, cdf[0] = 0.
//  cdf_min is the first non-zero value in cdf[], found on the host after the scan.
//
// ─────────────────────────────────────────────────────────────────────────────
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
