#include "cpu_sort.h"
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <ctime>

/* ------------------------------------------------------------------ */
/*  Timing helper                                                       */
/* ------------------------------------------------------------------ */
static double wall_ms()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

/* ================================================================== */
/*  std::sort wrapper                                                   */
/* ================================================================== */
float run_std_sort(int *arr, int n)
{
    double t0 = wall_ms();
    std::sort(arr, arr + n);
    return (float)(wall_ms() - t0);
}

/* ================================================================== */
/*  Recursive merge sort — pre-allocated temp buffer                   */
/*                                                                     */
/*  Using a single temp buffer (no malloc/free per merge) avoids the   */
/*  heap overhead that was present in the previous implementation.     */
/* ================================================================== */
static void merge_step(int *arr, int *tmp, int left, int mid, int right)
{
    /* merge arr[left..mid) and arr[mid..right) via tmp */
    int i = left, j = mid, k = left;
    while (i < mid && j < right)
        tmp[k++] = (arr[i] <= arr[j]) ? arr[i++] : arr[j++];
    while (i < mid)   tmp[k++] = arr[i++];
    while (j < right) tmp[k++] = arr[j++];
    memcpy(arr + left, tmp + left, (right - left) * sizeof(int));
}

static void merge_sort_rec(int *arr, int *tmp, int left, int right)
{
    if (right - left <= 1) return;
    int mid = left + (right - left) / 2;
    merge_sort_rec(arr, tmp, left, mid);
    merge_sort_rec(arr, tmp, mid,  right);
    merge_step    (arr, tmp, left, mid, right);
}

float run_cpu_merge_sort(int *arr, int n)
{
    int *tmp = (int *)malloc((size_t)n * sizeof(int));
    double t0 = wall_ms();
    merge_sort_rec(arr, tmp, 0, n);
    float ms = (float)(wall_ms() - t0);
    free(tmp);
    return ms;
}
