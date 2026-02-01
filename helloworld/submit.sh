#!/bin/bash
#PBS -N helloworld_gpu
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -l walltime=00:05:00
#PBS -q normal
#PBS -P personal

cd $PBS_O_WORKDIR

# Load CUDA module
module load cuda

# Run your program
./helloworld