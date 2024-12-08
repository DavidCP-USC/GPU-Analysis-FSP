#!/bin/sh
# SBATCH -n 1
# SBATCH -c 32
# SBATCH --mem 64G
# SBATCH --gres gpu:a100
# SBATCH -t 00:10:00
# 3480000000
#nvcc ./../../code/matrizMul-simple.cu -o ./../../code/matrizMul-simple
primerValor=32
for p in $(seq 2 2 32)
do
    nombreTrabajo="m-$primerValor-$p"
    sbatch -J $nombreTrabajo -o matrix.o -e matrix.e -n 1 --gres=gpu:a100 -c 32 --mem=64GB --time=01:59:00 task_VariasEjecuciones.sh $primerValor $p
done
