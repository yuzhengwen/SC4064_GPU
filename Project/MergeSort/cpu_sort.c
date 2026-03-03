#include <stdlib.h>
#include "cpu_sort.h"

static void cpu_merge(int *arr, int left, int mid, int right)
{
    int n1 = mid - left + 1, n2 = right - mid;
    int *L = (int *)malloc(n1 * sizeof(int));
    int *R = (int *)malloc(n2 * sizeof(int));
    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int j = 0; j < n2; j++) R[j] = arr[mid + 1 + j];
    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2)
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
    free(L);
    free(R);
}

void cpu_mergeSort(int *arr, int left, int right)
{
    if (left < right) {
        int mid = left + (right - left) / 2;
        cpu_mergeSort(arr, left, mid);
        cpu_mergeSort(arr, mid + 1, right);
        cpu_merge(arr, left, mid, right);
    }
}
