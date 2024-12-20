#!/bin/sh
# SBATCH -n 1
# SBATCH -c 32
# SBATCH --mem 64G
# SBATCH --gres gpu:a100
# SBATCH -t 00:10:00
# 3480000000
nvcc ./../../code/matrizMul-simple.cu -o ./../../code/matrizMul-simple
srun --gres=gpu:a100 -c 32 --mem=20G -t 00:20:00 ./../../code/matrizMul-simple ${1} ${2} ${3} ${4} ${5}
rm ./../../code/matrizMul-simple