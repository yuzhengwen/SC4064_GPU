#pragma once

/*
 * cpu_sort.h — CPU-side sorting routines
 *
 * Each function sorts arr[0..n) in-place and returns elapsed wall-clock
 * time in milliseconds.
 */

float run_std_sort      (int *arr, int n);   // std::sort  (introsort)
float run_cpu_merge_sort(int *arr, int n);   // recursive merge sort, pre-alloc temp
