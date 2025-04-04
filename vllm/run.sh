#!/bin/bash
#SBATCH --job-name=llm test      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=60       # Definition of cpu-cores per task
#SBATCH --partition=small_gpu8
#SBATCH --gres=gpu:5
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

python /beegfs/home/e/eilermas/Projekte/pais2025/main.py \
    --model_path /beegfs/home/e/eilermas/Projekte/pais2025/models \
    --output_dir /beegfs/home/e/eilermas/Projekte/pais2025/experiments\
    --output_file first_test.txt
       