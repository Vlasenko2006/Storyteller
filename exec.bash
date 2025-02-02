#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --job-name=tst
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=71:59:00
#SBATCH --account= #specify your account
#SBATCH --partition= #specify your partition
#SBATCH --error=tst_nn.out
#SBATCH -o nn.out

#SBATCH --exclusive                # https://slurm.schedmd.com/sbatch.html#OPT_exclusive
#SBATCH --mem=0                    # Request all memory available on all nodes


#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:#specify the path to anaconda library



srun #specify your path to python story_telling_nn.py #Lang5_self_long.py
