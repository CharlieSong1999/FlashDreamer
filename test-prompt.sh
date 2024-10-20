#!/bin/bash
#PBS -P kf09
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=32768MB
#PBS -l walltime=00:09:00
#PBS -l wd
#PBS -l storage=scratch/kf09

cd /scratch/kf09/lz1278/ANU-COMP8536-2024s2-main
export CONDA_ENV='/scratch/kf09/lz1278/lyenv/miniconda3/bin/activate'
source $CONDA_ENV /scratch/kf09/lz1278/lyenv/flash3d

module load cuda/12.2.2
python3 test_prompt.py --rotate_angle_list "-10" 2>&1 | tee ./test_prompt.txt