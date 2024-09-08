#!/bin/bash -l

#Slurm parameters
#SBATCH --job-name=segguiddiff
#SBATCH --output=segguiddiff%j.%N.out
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


# Activate everything you need

#conda activate /anaconda3/envs/myenv
module load cuda
pyenv activate myenv38 #myenv #venv
pip uninstall nvidia_cublas_cu11
nvcc --version
python -c "import torch; print(torch.__version__)"
export CUDA_LAUNCH_BLOCKING=1

# Run your python code

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --mode eval_many \
    --model_type DDIM \
    --img_size 256 \
    --num_img_channels 1 \
    --dataset "AutoPET" \
    --eval_batch_size 32 \
    --eval_sample_size 320 \
    --seg_dir "/data/private/autoPET/medicaldiffusion_results/test_results/ddpm/AutoPET/output_with_segconv_64out/video_results/mask" \
    --segmentation_guided \
    --num_segmentation_classes 37