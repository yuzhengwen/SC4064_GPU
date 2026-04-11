#!/bin/bash
#PBS -N merge_sort_benchmark
#PBS -l select=1:ncpus=4:ngpus=1:mem=4GB
#PBS -l walltime=00:45:00
#PBS -q normal
#PBS -j oe
#PBS -o merge_sort_benchmark.out
#PBS -P personal

cd "$PBS_O_WORKDIR"

echo "========================================"
echo "  Job:       $PBS_JOBID"
echo "  Node:      $(hostname)"
echo "  Started:   $(date)"
echo "  Workdir:   $PBS_O_WORKDIR"
echo "========================================"
echo ""

module purge
module load cuda

echo "--- CUDA version ---"
nvcc --version
echo ""

echo "--- GPU info ---"
nvidia-smi
echo ""

echo "========================================"
echo "  Compiling..."
echo "========================================"

nvcc -O3 -std=c++14 \
    main.cu          \
    input_gen.cu     \
    cpu_sort.cu      \
    gpu_naive.cu     \
    gpu_smem.cu      \
    gpu_bsearch.cu   \
    gpu_thrust.cu    \
    -o merge_sort_benchmark

if [ $? -ne 0 ]; then
    echo "Compilation FAILED"
    exit 1
fi
echo "Compilation OK"
echo ""

echo "========================================"
echo "  Running benchmark..."
echo "========================================"
./merge_sort_benchmark

echo ""
echo "========================================"
echo "  Completed: $(date)"
echo "========================================"
