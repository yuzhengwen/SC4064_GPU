#ifndef GPU_SMEM_H
#define GPU_SMEM_H

// Shared memory tiled GPU merge sort.
// Phase 1: each block sorts 1024 elements entirely in shared memory.
// Phase 2: standard global memory merges for widths >= 1024.
// Returns pointer to whichever device buffer holds the sorted result.
int *gpu_mergeSort_smem(int *d_data, int *d_temp, int n);

#endif
