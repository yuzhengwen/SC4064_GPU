#!/bin/bash
#PBS -N merge
#PBS -l select=1:ncpus=1:ngpus=1:mem=8GB
#PBS -l walltime=00:15:00
#PBS -q normal
#PBS -j oe
#PBS -o merge.out
#PBS -P personal

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
nvcc -O3 merge.cu -o merge
echo ""

# Run the program
echo "Running merge sort..."
echo "================================"
./merge
echo ""

# Job completion
echo "================================"
echo "Job completed on $(date)"