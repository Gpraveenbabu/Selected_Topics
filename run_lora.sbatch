#!/bin/bash

#SBATCH --job-name=lora_backdoor
#SBATCH --account=ag_gipp
#SBATCH --partition=scc-gpu
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=./slurm_files/slurm-%x-%j.out
#SBATCH --error=./slurm_files/slurm-%x-%j.err

source ~/.bashrc
conda activate lora_env

cd /mnt/vast-standard/home/p.geddada/u28147/projects/trojan/Selected_Topics/github_project

echo "=========================================="
echo "Starting LoRA backdoor experiment"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $PWD"
python --version
nvidia-smi

echo "=========================================="
echo "TRAINING"
echo "=========================================="

python -u backdoor_trigger.py \
    --model /mnt/vast-standard/home/p.geddada/u28147/Llama-3.2-3B-Instruct \
    --train-file data/pizza_train.json \
    --test-file data/pizza_test.json \
    --output-dir ./lora_final \
    --epochs 3 \
    --batch-size 1 \
    --lr 5e-5 \
    --no-chat

echo "=========================================="
echo "TRAINING FINISHED"
echo "=========================================="
date
