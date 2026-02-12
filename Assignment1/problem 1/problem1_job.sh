#!/bin/bash
#SBATCH --job-name=vec_add          # Job name
#SBATCH --partition=gpu             # Partition (queue) name
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks (processes)
#SBATCH --gres=gpu:1                # Number of GPUs per node
#SBATCH --time=00:15:00             # Time limit (HH:MM:SS)
#SBATCH --mem=16GB                  # Memory per node
#SBATCH --output=vec_add_%j.out     # Standard output file (%j = job ID)
#SBATCH --error=vec_add_%j.err      # Standard error file

# Print job information
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
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
make clean
make
echo ""

# Run the program
echo "Running vector addition..."
echo "================================"
./vector_add
echo ""

# Job completion
echo "================================"
echo "Job completed on $(date)"
