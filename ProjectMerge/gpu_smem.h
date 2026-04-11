#pragma once

/*
 * gpu_smem.h — Improvement 1: Shared-memory tile sort + global merge
 *
 * Phase 1 — each block loads TILE_SIZE elements into shared memory and
 *            sorts them entirely on-chip (bottom-up merge in smem).
 *            This eliminates the first log2(TILE_SIZE) global-memory passes.
 *
 * Phase 2 — iterative global-memory merge starting at width = TILE_SIZE,
 *            identical to the naive kernel.
 *
 * Returns total kernel time in ms (excludes H2D / D2H transfers).
 */

float run_gpu_smem(const int *h_src, int *h_dst, int n);
