#pragma once

/*
 * gpu_bsearch.h — Improvement 2: Shared-memory tile sort +
 *                                Parallel co-rank merge
 *
 * Phase 1 — same shared-memory tile sort as gpu_smem (TILE_SIZE tiles).
 *
 * Phase 2 — fully parallel merge using the co-rank algorithm.
 *            Each output position is assigned to its own thread.
 *            The thread uses binary search (O(log width)) to determine
 *            which of the two input runs contributes its element,
 *            eliminating the sequential bottleneck of the naive merge.
 *
 *   Naive/smem Phase 2: 1 thread  merges 2·width elements sequentially.
 *   BSearch   Phase 2: N threads each do O(log width) work in parallel.
 *
 * Returns total kernel time in ms (excludes H2D / D2H transfers).
 */

float run_gpu_bsearch(const int *h_src, int *h_dst, int n);
