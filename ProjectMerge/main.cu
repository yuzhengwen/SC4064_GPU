/*
 * main.cu — Merge Sort GPU Benchmark Coordinator
 *
 * Algorithms compared
 * -------------------
 *  CPU  1. std::sort            (C++ introsort)
 *  CPU  2. Recursive merge sort (pre-allocated temp buffer)
 *  GPU  3. Naive merge sort     (global memory only)
 *  GPU  4. SMEM merge sort      (Improvement 1: shared-memory tile sort)
 *  GPU  5. BSearch merge sort   (Improvement 2: parallel co-rank merge)
 *  GPU  6. thrust::sort         (library baseline)
 *
 * Sizes: 2^10, 2^15, 2^20, 2^25
 *
 * Output format
 * -------------
 *  Per size: per-pass timing table for each GPU variant,
 *            followed by a correctness check.
 *  End:      summary table (time, throughput, bandwidth, speedup).
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "cuda_check.h"
#include "input_gen.h"
#include "cpu_sort.h"
#include "gpu_naive.h"
#include "gpu_smem.h"
#include "gpu_bsearch.h"
#include "gpu_thrust.h"

/* ------------------------------------------------------------------ */
/*  Verification                                                        */
/* ------------------------------------------------------------------ */
static bool verify(const int *ref, const int *result, int n)
{
    for (int i = 0; i < n; i++)
        if (ref[i] != result[i]) return false;
    return true;
}

/* ------------------------------------------------------------------ */
/*  Separator helpers                                                   */
/* ------------------------------------------------------------------ */
static void sep(char c = '-', int w = 78) {
    for (int i = 0; i < w; i++) putchar(c);
    putchar('\n');
}

/* ================================================================== */
/*  Result record for the summary table                                */
/* ================================================================== */
struct Record {
    const char *label;
    float times[4];   /* one per size index */
};

static const int SIZES[]  = { 1<<10, 1<<15, 1<<20, 1<<25 };
static const int EXPS[]   = { 10, 15, 20, 25 };
static const int N_SIZES  = 4;
static const int N_ALGOS  = 6;

/* ================================================================== */
/*  main                                                                */
/* ================================================================== */
int main()
{
    /* ---- allocate worst-case buffers once ---- */
    int max_n = SIZES[N_SIZES - 1];
    size_t max_bytes = (size_t)max_n * sizeof(int);

    int *h_orig = (int *)malloc(max_bytes);
    int *h_work = (int *)malloc(max_bytes);
    int *h_ref  = (int *)malloc(max_bytes);   /* std::sort reference */
    int *h_gpu  = (int *)malloc(max_bytes);

    if (!h_orig || !h_work || !h_ref || !h_gpu) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    /* ---- result table ---- */
    Record records[N_ALGOS];
    records[0].label = "CPU std::sort";
    records[1].label = "CPU merge sort";
    records[2].label = "GPU naive";
    records[3].label = "GPU smem";
    records[4].label = "GPU bsearch";
    records[5].label = "GPU Thrust";

    printf("\n");
    sep('=');
    printf("  Merge Sort GPU Benchmark\n");
    printf("  Sizes: 2^10, 2^15, 2^20, 2^25\n");
    sep('=');

    /* ==================================================================
     *  Main loop over array sizes
     * ================================================================== */
    for (int si = 0; si < N_SIZES; si++) {
        int    n     = SIZES[si];
        size_t bytes = (size_t)n * sizeof(int);

        double tp_scale = n / 1e6;      /* Melem/s divisor */
        double bw_scale = 2.0 * bytes / 1e9;  /* GB per pass (read+write) */

        /* Generate random input once per size */
        gen_random(h_orig, n, 42);

        printf("\n");
        sep('=');
        printf("  N = 2^%d = %d elements  (%.1f MB per array)\n",
               EXPS[si], n, bytes / 1e6);
        sep('=');

        /* ---- Build reference with std::sort ---- */
        memcpy(h_ref, h_orig, bytes);
        float ref_ms = run_std_sort(h_ref, n);   /* h_ref is now sorted */

        /* ---- 1. std::sort ---- */
        memcpy(h_work, h_orig, bytes);
        float t_std = run_std_sort(h_work, n);
        records[0].times[si] = t_std;
        printf("  [1] std::sort         : %.4f ms  (%s)\n",
               t_std, verify(h_ref, h_work, n) ? "PASSED" : "FAILED");

        /* ---- 2. CPU merge sort ---- */
        memcpy(h_work, h_orig, bytes);
        float t_cpu = run_cpu_merge_sort(h_work, n);
        records[1].times[si] = t_cpu;
        printf("  [2] CPU merge sort    : %.4f ms  (%s)\n",
               t_cpu, verify(h_ref, h_work, n) ? "PASSED" : "FAILED");

        /* ---- 3. GPU naive ---- */
        printf("\n  [3] GPU Naive\n");
        printf("    %-5s  %-14s  %-12s  %-12s  %-12s  %-10s\n",
               "Pass", "Width", "Merges", "Time(ms)", "Elem/s", "BW(GB/s)");
        printf("    %s\n",
               "-----------------------------------------------------------------------");
        float t_naive = run_gpu_naive(h_orig, h_gpu, n);
        records[2].times[si] = t_naive;
        printf("  Correct: %s\n", verify(h_ref, h_gpu, n) ? "PASSED" : "FAILED");

        /* ---- 4. GPU smem ---- */
        printf("\n  [4] GPU Smem (Improvement 1: shared-memory tile sort)\n");
        printf("    %-5s  %-14s  %-12s  %-12s  %-12s  %-10s\n",
               "Phase", "Width/Desc", "Blocks", "Time(ms)", "Elem/s", "BW(GB/s)");
        printf("    %s\n",
               "-----------------------------------------------------------------------");
        float t_smem = run_gpu_smem(h_orig, h_gpu, n);
        records[3].times[si] = t_smem;
        printf("  Correct: %s\n", verify(h_ref, h_gpu, n) ? "PASSED" : "FAILED");

        /* ---- 5. GPU bsearch ---- */
        printf("\n  [5] GPU BSearch (Improvement 2: parallel co-rank merge)\n");
        printf("    %-5s  %-14s  %-12s  %-12s  %-12s  %-10s\n",
               "Phase", "Width/Desc", "Blocks", "Time(ms)", "Elem/s", "BW(GB/s)");
        printf("    %s\n",
               "-----------------------------------------------------------------------");
        float t_bsearch = run_gpu_bsearch(h_orig, h_gpu, n);
        records[4].times[si] = t_bsearch;
        printf("  Correct: %s\n", verify(h_ref, h_gpu, n) ? "PASSED" : "FAILED");

        /* ---- 6. Thrust ---- */
        float t_thrust = run_thrust_sort(h_orig, h_gpu, n);
        records[5].times[si] = t_thrust;
        printf("\n  [6] Thrust::sort      : %.4f ms  (%s)\n",
               t_thrust, verify(h_ref, h_gpu, n) ? "PASSED" : "FAILED");
    }

    /* ==================================================================
     *  Summary table
     * ================================================================== */
    printf("\n\n");
    sep('=');
    printf("  SUMMARY TABLE — Time (ms)\n");
    sep('=');

    /* Header row */
    printf("  %-20s", "Algorithm");
    for (int si = 0; si < N_SIZES; si++)
        printf("  %10s", "");
    printf("\n");

    printf("  %-20s", "");
    for (int si = 0; si < N_SIZES; si++)
        printf("  %9s ", "");
    printf("\n");

    /* Column headers with sizes */
    printf("  %-22s", "Algorithm \\ Size");
    for (int si = 0; si < N_SIZES; si++)
        printf("  %8s  ", "--------");
    printf("\n");

    printf("  %-22s", "");
    for (int si = 0; si < N_SIZES; si++) {
        char buf[16];
        snprintf(buf, sizeof(buf), "2^%d", EXPS[si]);
        printf("  %8s  ", buf);
    }
    printf("\n");
    sep('-');

    for (int ai = 0; ai < N_ALGOS; ai++) {
        printf("  %-22s", records[ai].label);
        for (int si = 0; si < N_SIZES; si++)
            printf("  %8.3f  ", records[ai].times[si]);
        printf("\n");
    }
    sep('-');

    /* Throughput table (Melem/s) */
    printf("\n");
    sep('=');
    printf("  SUMMARY TABLE — Throughput (Melem/s)\n");
    sep('=');

    printf("  %-22s", "");
    for (int si = 0; si < N_SIZES; si++) {
        char buf[16];
        snprintf(buf, sizeof(buf), "2^%d", EXPS[si]);
        printf("  %8s  ", buf);
    }
    printf("\n");
    sep('-');

    for (int ai = 0; ai < N_ALGOS; ai++) {
        printf("  %-22s", records[ai].label);
        for (int si = 0; si < N_SIZES; si++) {
            double tp = SIZES[si] / (records[ai].times[si] / 1000.0) / 1e6;
            printf("  %8.2f  ", tp);
        }
        printf("\n");
    }
    sep('-');

    /* Speedup vs CPU merge sort */
    printf("\n");
    sep('=');
    printf("  SPEEDUP vs CPU merge sort\n");
    sep('=');

    printf("  %-22s", "");
    for (int si = 0; si < N_SIZES; si++) {
        char buf[16];
        snprintf(buf, sizeof(buf), "2^%d", EXPS[si]);
        printf("  %8s  ", buf);
    }
    printf("\n");
    sep('-');

    for (int ai = 0; ai < N_ALGOS; ai++) {
        printf("  %-22s", records[ai].label);
        for (int si = 0; si < N_SIZES; si++) {
            double speedup = records[1].times[si] / records[ai].times[si];
            printf("  %8.2fx ", speedup);
        }
        printf("\n");
    }
    sep('-');

    printf("\nDone.\n");

    free(h_orig);
    free(h_work);
    free(h_ref);
    free(h_gpu);
    return 0;
}
