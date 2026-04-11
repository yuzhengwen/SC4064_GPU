#include "input_gen.h"
#include <cstdlib>

void gen_random(int *arr, int n, unsigned int seed)
{
    srand(seed);
    for (int i = 0; i < n; i++)
        arr[i] = rand() % 1000000;
}

void gen_sorted(int *arr, int n)
{
    for (int i = 0; i < n; i++)
        arr[i] = i;
}

void gen_reverse(int *arr, int n)
{
    for (int i = 0; i < n; i++)
        arr[i] = n - 1 - i;
}
