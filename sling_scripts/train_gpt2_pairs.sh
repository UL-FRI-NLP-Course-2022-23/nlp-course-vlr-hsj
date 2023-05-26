#!/bin/bash
#SBATCH --job-name=FT-GPTP
#SBATCH --output=logs/sling-GPT2P-%J.out
#SBATCH --error=logs/sling-GPT2P-%J.err
#SBATCH --time=1-00:00:00 # job time limit - full format is D-H:M:S
#SBATCH --nodes=1 # number of nodes
#SBATCH --gres=gpu:2 # number of gpus
#SBATCH --ntasks=1 # number of tasks
#SBATCH --mem-per-gpu=24G # memory allocation
#SBATCH --partition=gpu # partition to run on nodes that contain gpus
#SBATCH --cpus-per-task=12 # number of allocated cores

source /d/hpc/projects/FRI/vh0153/miniconda3/etc/profile.d/conda.sh # intialize conda
conda activate nlp
srun --nodes=1 --exclusive --gres=gpu:2 --ntasks=1 python /d/hpc/projects/FRI/vh0153/nlp-course-vlr-hsj/scripts/train_gpt_pairs.py