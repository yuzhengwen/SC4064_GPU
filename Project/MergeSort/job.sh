#!/bin/bash
#PBS -N merge_benchmark
#PBS -l select=1:ncpus=1:ngpus=1:mem=512MB
#PBS -l walltime=00:30:00
#PBS -q normal
#PBS -j oe
#PBS -o merge_benchmark.out
#PBS -P personal

cd $PBS_O_WORKDIR

echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $PBS_JOBID"
echo "Working directory: $PBS_O_WORKDIR"
echo "================================"
echo ""

module purge
module load cuda

echo "CUDA Version:"
nvcc --version
echo ""

echo "GPU Information:"
nvidia-smi
echo ""
echo "================================"
echo ""

echo "Compiling..."
nvcc -O3 -std=c++14 \
    main.cu \
    cpu_sort.c \
    gpu_naive.cu \
    gpu_smem.cu \
    gpu_thrust.cu \
    -o merge_benchmark
echo "Compilation done."
echo ""

echo "Running merge sort benchmark..."
echo "================================"
./merge_benchmark
echo ""

echo "================================"
echo "Job completed on $(date)"
