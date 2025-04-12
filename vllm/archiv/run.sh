#!/bin/bash

# PyTorch distributed setup
export MASTER_PORT=29500
echo "MASTER_PORT="$MASTER_PORT

export RANK=$SLURM_PROCID
echo "RANK="$RANK

export WORLD_SIZE=$((1 * 1))
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

export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_HOST_IP=127.0.0.1

python main.py \
     --model_path ~/vllm_models/models/gemma-3-27b-it \
     --output_dir ~/experiments \
     --output_file test.txt \
     --model gemma_3_27b_it \
     --tensor_parallel_size 2 \
