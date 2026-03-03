#ifndef GPU_NAIVE_H
#define GPU_NAIVE_H

// Naive GPU merge sort — global memory only.
// One thread per merge operation per pass.
// Prints per-pass timing to stdout.
// Returns pointer to whichever device buffer holds the sorted result.
int *gpu_mergeSort_naive(int *d_data, int *d_temp, int n);

#endif
