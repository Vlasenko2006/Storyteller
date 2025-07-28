#!/bin/bash

#SBATCH --job-name=bert
#SBATCH --nodes=3                # <-- Change this to 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --time=71:59:00
#SBATCH --account=ksm
#SBATCH --partition=pGPU
#SBATCH --error=e-wiki_bert.out
#SBATCH --output=wiki_bert.out
#SBATCH --exclusive
#SBATCH --mem=0

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$((10000 + RANDOM % 50000))

export MASTER_ADDR MASTER_PORT

srun /gpfs/work/vlasenko/07/NN/fatenv/gpt2_finetuning_env/bin/python3.10 story_telling_nn.py --ddp
