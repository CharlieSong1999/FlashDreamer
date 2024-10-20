#!/bin/bash
#PBS -P kf09
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=32768MB
#PBS -l walltime=00:19:00
#PBS -l wd
#PBS -l storage=scratch/kf09

cd /scratch/kf09/lz1278/ANU-COMP8536-2024s2-main
export CONDA_ENV='/scratch/kf09/lz1278/lyenv/miniconda3/bin/activate'
source $CONDA_ENV /scratch/kf09/lz1278/lyenv/flash3d

module load cuda/12.2.2

# python3 ./flash3d/generator.py
# python3 ./pre-process-img.py --output_height 512 --output_width 512
python3 ./vlm_diffusion_pipeline.py --image_path './resize_output/flash3drender_test_resized.png' \
        --mask_path  './resize_output/flash3drender_test_mask_resized.png' \
        --base_model 'stable-diffusion-xl' \
        --prompt_diffusion 'A indoor scene, a room, a window, two sofas.' \
        | tee ./test_ppl_logs.txt


