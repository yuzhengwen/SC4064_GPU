// simple merge sort implementation in C
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

void merge(int arr[], int left, int mid, int right)
{
    int i, j, k;
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // create temp arrays
    int *L = (int *)malloc(n1 * sizeof(int));
    int *R = (int *)malloc(n2 * sizeof(int));

    // copy data to temp arrays
    for (i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    // merge the temp arrays back into arr[left..right]
    i = 0;    // initial index of first sub-array
    j = 0;    // initial index of second sub-array
    k = left; // initial index of merged sub-array

    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
        {
            arr[k] = L[i];
            i++;
        }
        else
        {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    // copy the remaining elements of L[], if there are any
    while (i < n1)
    {
        arr[k] = L[i];
        i++;
        k++;
    }

    // copy the remaining elements of R[], if there are any
    while (j < n2)
    {
        arr[k] = R[j];
        j++;
        k++;
    }

    // free the temp arrays
    free(L);
    free(R);
}

void mergeSort(int arr[], int left, int right)
{
    if (left < right)
    {
        // find the middle point
        int mid = left + (right - left) / 2;

        // sort first and second halves
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);

        // merge the sorted halves
        merge(arr, left, mid, right);
    }
}

// Error checking macro
#define CUDA_CHECK(call)                                                                                      \
    do                                                                                                        \
    {                                                                                                         \
        cudaError_t err = call;                                                                               \
        if (err != cudaSuccess)                                                                               \
        {                                                                                                     \
            fprintf(stderr, "CUDA error: %s (at line %s:%d)\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(err);                                                                                        \
        }                                                                                                     \
    } while (0)

int main()
{
    // create a large array of random integers
    int N = 1 << 20;
    int *arr = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++)
    {
        arr[i] = rand() % 1000000; // random integers between 0 and 999999
    }

    // sort using merge sort and benchmark
    printf("Sorting array of %d elements using merge sort...\n", N);
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    mergeSort(arr, 0, N - 1);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Merge sort took %.4f seconds\n", elapsed);

    printf("\nDone!\n");
    return 0;
}
