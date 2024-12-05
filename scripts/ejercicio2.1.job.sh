#!/bin/bash

#SBATCH -c 32
#SBATCH --mem 247G
#SBATCH --gres gpu:a100:1
#SBATCH -t 02:00:00
#SBATCH -J P3_2_1
#SBATCH -o ./outs/P3_2_1.o
#SBATCH -e ./outs/P3_2_1.e

rm ./data/P3_2_1.txt

nvcc ./code/vectorAdd.cu -o vectorAdd_1.o

start=2700000000
end=3480000000
step=10000000

for ((i=$start; i<=$end; i+=$step)); do
	srun vectorAdd_1.o $i >> ./data/P3_2_1.txt
done

rm vectorAdd_1.o
