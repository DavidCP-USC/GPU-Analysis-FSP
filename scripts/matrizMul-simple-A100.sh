#!/bin/sh
# SBATCH -n 1
# SBATCH -c 32
# SBATCH --mem 64G
# SBATCH --gres gpu:a100
# SBATCH -t 00:10:00
# 3480000000
nvcc ./../code/matrizMul-simple.cu -o ./../code/matrizMul-simple
srun --gres=gpu:a100 -c 32 --mem=64G -t 1 ./../code/matrizMul-simple ${1} ${2} ${3}
rm ./../code/matrizMul-simple