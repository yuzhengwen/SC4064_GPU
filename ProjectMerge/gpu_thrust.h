#pragma once

/*
 * gpu_thrust.h — Library baseline: thrust::sort
 *
 * Allocates device memory, copies h_src → device, sorts with
 * thrust::sort, copies result → h_dst.
 *
 * Returns pure sort kernel time in ms (H2D and D2H excluded from
 * the reported time so it is directly comparable to the other GPU
 * variants).
 */

float run_thrust_sort(const int *h_src, int *h_dst, int n);
