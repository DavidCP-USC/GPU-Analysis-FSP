#!/bin/bash

#SBATCH -c 32
#SBATCH --mem 247G
#SBATCH --gres gpu:a100:1
#SBATCH -t 02:00:00
#SBATCH -J P3_2_4
#SBATCH -o ./outs/P3_2_4.o
#SBATCH -e ./outs/P3_2_4.e

rm ./data/P3_2_4.txt

nvcc ./code/vectorAdd_2.cu -o vectorAdd_4.o

start=2700000000
end=3480000000
step=10000000

for ((i=$start; i<=$end; i+=$step)); do
	srun vectorAdd_4.o $i >> ./data/P3_2_4.txt
done

rm vectorAdd_4.o
