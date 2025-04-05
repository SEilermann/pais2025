#!/bin/bash
#SBATCH --job-name=llm_test      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=72
#SBATCH --partition=small_gpu8
#SBATCH --gpus-per-task=8
#SBATCH --time=02:30:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=sebastian.eilermann@hsu-hh.de
#SBATCH --output=slurmjob%j.log 
 
module purge
module load cuda/12.4.1
module load gcc/13.2.0
module load anaconda3/2021.11
eval "$(conda shell.bash hook)"
conda info --envs
conda activate paisenv

echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# --- NCCL Performance Tuning ---
#export NCCL_DEBUG=INFO
#export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=ib0
#export NCCL_NET_GDR_LEVEL=SYS
#export NCCL_P2P_LEVEL=SYS

# Optional: other envs
#export TORCH_DISTRIBUTED_DEBUG=INFO



#Qwen2_5_72B, DeepSeek_R1_Distill_Llama_70B Llama_3_1_8B

srun python /beegfs/home/e/eilermas/Projekte/pais2025/vllm/main.py \
    --model_path /beegfs/home/e/eilermas/Projekte/models \
    --output_dir /beegfs/home/e/eilermas/Projekte/pais2025/experiments \
    --output_file first_test_Llama_3_1_8B.txt \
    --model DeepSeek_R1_Distill_Llama_70B \
    --tensor_parallel_size 8 \
    

