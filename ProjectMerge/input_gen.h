#pragma once

/*
 * input_gen.h — random / patterned array generators
 *
 * All functions write `n` integers into the caller-allocated array `arr`.
 * `seed` is used for reproducible random sequences.
 */

void gen_random (int *arr, int n, unsigned int seed = 42);
void gen_sorted (int *arr, int n);
void gen_reverse(int *arr, int n);
