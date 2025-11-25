#!/bin/bash -l
#PBS -A FASTBAYES
#PBS -l select=1:ngpus=4
#PBS -l walltime=23:59:00         
#PBS -N base_vae
#PBS -q gpu
#PBS -j n
#PBS -m be


# Print job information
echo "Working directory: $PBS_O_WORKDIR"
cd $PBS_O_WORKDIR

echo "Job ID: $PBS_JOBID"
echo "Running on host: $(hostname)"
echo "Allocated nodes:"
cat $PBS_NODEFILE

# Load necessary modules
module load gcc openmpi    # Ensure OpenMPI is loaded
module list                # List loaded modules
which mpirun               # Verify mpirun location

# Source the conda environment
source /soft/anaconda3/2020.02/etc/profile.d/conda.sh
conda activate /lcrc/project/FastBayes/Anirban_VI/PyTorchTF-GPU_AB


# Change to the directory where the job is submitted from
cd $PBS_O_WORKDIR

# Define the master address and port for DDP
MASTER_ADDR=$(hostname)  # Using the current node as the master address
MASTER_PORT=12347        # Fixed port; ensure it's free on the system

# Define the number of GPUs on the node
NUM_GPUS=$(nvidia-smi -L | wc -l)

# Run the Python script using torchrun with the specified arguments
torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS vae_cifar10_ddp_medium.py \
  --model "icfm" \
  --output_dir "/lcrc/project/FastBayes/Anirban_VI/Diffusion_models/code/Latent_CFM/code/darcy_flow/runs/" \
  --lr 2e-4 \
  --ema_decay 0.9999 \
  --batch_size 128 \
  --num_workers 4 \
  --total_steps 100001 \
  --save_step 10000 \
  --parallel True \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \


wait








