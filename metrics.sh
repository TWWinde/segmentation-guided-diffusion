#!/bin/bash -l

#Slurm parameters
#SBATCH --job-name=metrics
#SBATCH --output=metrics%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-23:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --qos=batch
#SBATCH --nodelist=linse21
# SBATCH --qos=shortbatch
# SBATCH --partition=highperf
#SBATCH --gpus=rtx_a5000:1
#SBATCH --nodelist=linse19




python metrics.py
