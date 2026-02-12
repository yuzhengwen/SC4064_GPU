#!/bin/bash
#PBS -N mat_add
#PBS -l select=1:ncpus=1:ngpus=1:mem=16GB
#PBS -l walltime=00:15:00
#PBS -q gpu
#PBS -j oe
#PBS -o mat_add.out

cd $PBS_O_WORKDIR

echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $PBS_JOBID"
echo "================================"
echo ""

module purge
module load cuda/11.6

echo "CUDA Version:"
nvcc --version
echo ""

echo "GPU Information:"
nvidia-smi
echo ""
echo "================================"
echo ""

echo "Compiling..."
nvcc -O3 matrix_add.cu -o matrix_add
echo ""

echo "Running matrix addition..."
echo "================================"
./matrix_add
echo ""

echo "================================"
echo "Job completed on $(date)"
