#!/bin/bash
#SBATCH --job-name=ddp-vllm          # Job name
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --partition=small_gpu        # Partition name
#SBATCH --ntasks-per-node=2          # Number of tasks (processes) per node
#SBATCH --cpus-per-task=32            # CPU cores per task
#SBATCH --gres=gpu:2                 # Number of GPUs per node
#SBATCH --time=01:10:00              # Max run time
#SBATCH --mail-type=begin,end        # Email notifications
#SBATCH --mail-user=sebastian.eilermann@hsu-hh.de
#SBATCH --output=slurmjob%j.log      # Output log file

# Load required modules
module purge
module load nvhpc/24.3
module load miniforge3/24.3.0-0

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate new_env

export NCCL_ROOT="$NVHPC_ROOT/Linux_x86_64/24.3/comm_libs/nccl"
export CUDA_HOME="$(nvfortran -cuda -printcudaversion 2>&1 | grep "CUDA Path" | cut -d "=" -f 2)"

# Sanity checks
echo "========== CUDA ENV CHECK =========="
echo "CUDA_HOME = $CUDA_HOME"
which nvcc
nvcc --version
ls $CUDA_HOME/include/cuda.h
echo "====================================="

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
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV

export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0

export VLLM_HOST_IP=127.0.0.1



