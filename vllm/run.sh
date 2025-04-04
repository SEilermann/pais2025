#!/bin/bash
#SBATCH --job-name=llm_test      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=60       # Definition of cpu-cores per task
#SBATCH --partition=small_gpu8
#SBATCH --gres=gpu:8
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

python /beegfs/home/e/eilermas/Projekte/pais2025/vllm/main.py \
    --model_path /beegfs/home/e/eilermas/Projekte/models \
    --output_dir /beegfs/home/e/eilermas/Projekte/pais2025/experiments\
    --output_file first_test_deepseek_bigsmal.txt \
    --model DeepSeek_R1_Distill_Llama_70B \
    --tensor-parallel-size 8
       
