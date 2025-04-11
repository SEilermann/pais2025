#!/bin/bash
#SBATCH --job-name=ddp-vllm          # Job name
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --partition=small_gpu8        # Partition name
#SBATCH --ntasks-per-node=2          # Number of tasks (processes) per node
#SBATCH --cpus-per-task=8            # CPU cores per task
#SBATCH --mem=32G                    # Memory per node
#SBATCH --gres=gpu:2                 # Number of GPUs per node
#SBATCH --time=00:10:00              # Max run time
#SBATCH --mail-type=begin,end        # Email notifications
#SBATCH --mail-user=sebastian.eilermann@hsu-hh.de
#SBATCH --output=slurmjob%j.log      # Output log file

# Load required modules
module purge
module load USER-SPACK/0.22.1
module load gcc/13.2.0
module load cuda/12.4.1
module load anaconda3/2021.11

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate new_env

# Set CUDA_HOME from nvcc path
#export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
#export PATH="$CUDA_HOME/bin:$PATH"
#export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Sanity checks
echo "========== CUDA ENV CHECK =========="
echo "CUDA_HOME = $CUDA_HOME"
which nvcc
nvcc --version
ls $CUDA_HOME/include/cuda.h
echo "====================================="

export LD_LIBRARY_PATH=/cluster/spack/0.22.1/opt/spack/linux-rocky8-icelake/gcc-13.2.0/cuda-12.4.1-ge4bge4tgmhpkvd4bhxa6x6ir52bn7d6/lib64:$LD_LIBRARY_PATH

# PyTorch distributed setup
export MASTER_PORT=29500
echo "MASTER_PORT="$MASTER_PORT

export RANK=$SLURM_PROCID
echo "RANK="$RANK


export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export LOCAL_RANK=$SLURM_LOCALID

# NCCL and CUDA runtime debugging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV

export NCCL_SOCKET_IFNAME=lo

echo '####################'
echo "=== Testing NCCL availability in PyTorch ==="
python -c "import torch; print('NCCL available:', torch.distributed.is_nccl_available())"
echo '####################'

export NCCL_P2P_DISABLE=1

# Launch script (adjust path as needed)
srun python /beegfs/home/e/eilermas/Projekte/pais2025/vllm/test.py
#torchrun --nproc-per-node=2 /beegfs/home/e/eilermas/Projekte/pais2025/vllm/test.py

if [[ "$SLURM_PROCID" -eq 0 ]]; then
    rm -f master_port.txt
fi

