#!/bin/bash
#PBS -N mat_mul
#PBS -l select=1:ncpus=1:ngpus=1:mem=16GB
#PBS -l walltime=00:20:00
#PBS -q normal
#PBS -j oe
#PBS -o mat_mul.out
#PBS -P personal

cd $PBS_O_WORKDIR

# some basic info
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $PBS_JOBID"

module purge
module load cuda

echo "CUDA Version:"
nvcc --version
echo ""

echo "GPU Information:"
nvidia-smi
echo "================================"

nvcc -O3 matrix_mul.cu -o matrix_mul

echo "Running matrix multiplication exec..."
./matrix_mul
echo ""
echo "Job completed on $(date)"
