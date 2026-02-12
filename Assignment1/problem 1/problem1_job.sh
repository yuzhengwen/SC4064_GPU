#!/bin/bash
#PBS -N vec_add
#PBS -l select=1:ncpus=1:ngpus=1:mem=8GB
#PBS -l walltime=00:15:00
#PBS -q normal
#PBS -j oe
#PBS -o vec_add.out

# Change to the directory where job was submitted
cd $PBS_O_WORKDIR

# Print job information
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $PBS_JOBID"
echo "Working directory: $PBS_O_WORKDIR"
echo "================================"
echo ""

# Load required modules
module purge
module load cuda

# Print CUDA information
echo "CUDA Version:"
nvcc --version
echo ""

echo "GPU Information:"
nvidia-smi
echo ""
echo "================================"
echo ""

# Compile the program
echo "Compiling..."
nvcc -O3 vector_add.cu -o vector_add
echo ""

# Run the program
echo "Running vector addition..."
echo "================================"
./vector_add
echo ""

# Job completion
echo "================================"
echo "Job completed on $(date)"