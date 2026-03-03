#ifndef GPU_THRUST_H
#define GPU_THRUST_H

// Thrust sort (NVIDIA's optimised radix sort under the hood).
// Allocates its own device memory internally.
// Copies sorted result back into h_dst on return.
// Returns GPU sort time in milliseconds (excludes H2D/D2H transfer).
float thrust_sort_timed(int *h_src, int *h_dst, int n);

#endif
