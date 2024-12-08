#!/bin/sh
# SBATCH -n 1
# SBATCH -c 32
# SBATCH --mem 64G
# SBATCH --gres gpu:a100
# SBATCH -t 00:10:00
# 3480000000

for p in $(seq 200 200 5000)
do
    srun ./../../code/matrizMul-simple $p $p $p ${1} ${2}
done    