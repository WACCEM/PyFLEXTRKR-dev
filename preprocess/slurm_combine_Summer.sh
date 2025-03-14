#!/bin/bash
#SBATCH -A m1867
#SBATCH -J Summer
#SBATCH -t 00:30:00
#SBATCH -q debug
#SBATCH -C cpu
##SBATCH --nodes=2
##SBATCH --ntasks-per-node=128
#SBATCH -N 2
#SBATCH -c 128
#SBATCH --exclusive
#SBATCH --output=log_combine_Summer.log
#SBATCH --mail-type=END
#SBATCH --mail-user=zhe.feng@pnnl.gov
date
module load taskfarmer
export THREADS=128
cd $SCRATCH/preprocess
runcommands.sh task_combine_Summer.txt
date