#!/bin/bash
#SBATCH --job-name=ddp-torch     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --partition=small_gpu8
#SBATCH --ntasks-per-node=4      # total number of tasks per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gpus-per-node=4             # number of allocated gpus per node
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=sebastian.eilermann@hsu-hh.de
#SBATCH --output=slurmjob%j.log

# Function to find a free port
get_free_port() {
    python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()'
}

# Shared port setup
if [[ "$SLURM_PROCID" -eq 0 ]]; then
    export MASTER_PORT=$(get_free_port)
    echo $MASTER_PORT > master_port.txt
else
    while [ ! -f master_port.txt ]; do sleep 1; done
    export MASTER_PORT=$(cat master_port.txt)
fi

echo "MASTER_PORT=$MASTER_PORT"



export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
export NCCL_P2P_DISABLE=1
module purge
module load cuda/12.4.1
module load anaconda3/2021.11
module load anaconda3/2021.11
eval "$(conda shell.bash hook)"
conda info --envs
conda activate new_env


NCCL_DEBUG=TRACE torchrun --nproc-per-node=4 /beegfs/home/e/eilermas/Projekte/pais2025/vllm/main.py \
    --model_path /beegfs/home/e/eilermas/Projekte/models \
    --output_dir /beegfs/home/e/eilermas/Projekte/pais2025/experiments \
    --output_file first_test_Llama_3_3_70B.txt \
    --model Llama_3_3_70B_Instruct \
    --tensor_parallel_size 4 \
