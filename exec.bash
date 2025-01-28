#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --job-name=tst
#SBATCH --nodes=1 # unfortunately 3 is the max on strand at the moment. 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=71:59:00
#SBATCH --account=ksm
#SBATCH --partition=pGPU
#SBATCH --error=tst_nn.out
#SBATCH -o nn.out

#SBATCH --exclusive                # https://slurm.schedmd.com/sbatch.html#OPT_exclusive
#SBATCH --mem=0                    # Request all memory available on all nodes


#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfs/home/vlasenko/miniconda3/lib/



srun /gpfs/home/vlasenko/miniconda3/envs/gpuenv2/bin/python story_telling_nn.py #Lang5_self_long.py
