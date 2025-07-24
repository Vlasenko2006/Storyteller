#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --job-name=wiki
#SBATCH --nodes=1 # unfortunately 3 is the max on strand at the moment. 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=71:59:00
#SBATCH --account=ksm
#SBATCH --partition=pGPU
#SBATCH --error=e-wiki_lr.out
#SBATCH -o wiki_lr.out

#SBATCH --exclusive                # https://slurm.schedmd.com/sbatch.html#OPT_exclusive
#SBATCH --mem=0                    # Request all memory available on all nodes


#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfs/home/vlasenko/miniconda3/lib/




srun /gpfs/work/vlasenko/07/NN/fatenv/gpt2_finetuning_env/bin/python3.10 story_telling_nn.py #specify your path to python story_telling_nn.py 
