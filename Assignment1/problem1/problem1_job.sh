#!/bin/bash
#PBS -N vec_add
#PBS -l select=1:ncpus=1:ngpus=1:mem=8GB
#PBS -l walltime=00:15:00
#PBS -q normal
#PBS -j oe
#PBS -o vec_add.out
#PBS -P personal

# cd to working dir of submission
cd $PBS_O_WORKDIR

# some basic info
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $PBS_JOBID"
echo "Working directory: $PBS_O_WORKDIR"
echo "================================"
echo ""

# load CUDA
module purge
module load cuda

# check version
echo "CUDA Version:"
nvcc --version
echo ""

echo "GPU Information:"
nvidia-smi
echo "================================"
echo ""

# compile with CUDA compiler
echo "Compiling..."
# add flag for aggressive optimization
nvcc -O3 vector_add.cu -o vector_add
echo ""

# run prog
echo "Running vector addition executable..."
echo "================================"
./vector_add
echo ""

# complete
echo ""
echo "================================"
echo "Job completed on $(date)"