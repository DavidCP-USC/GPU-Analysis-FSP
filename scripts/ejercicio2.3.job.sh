#!/bin/bash

#SBATCH -c 32
#SBATCH --mem 247G
#SBATCH --gres gpu:a100:1
#SBATCH -t 02:00:00
#SBATCH -J P3_2_3
#SBATCH -o ./outs/P3_2_3.o
#SBATCH -e ./outs/P3_2_3.e

rm ./data/P3_2_3.txt

nvcc ./code/vectorAdd.cu -o vectorAdd_3.o

start=1
end=100
step=1

for ((i=$start; i<=$end; i+=$step)); do
	srun vectorAdd_3.o 3480000000 544 $i >> ./data/P3_2_3.txt
done

rm vectorAdd_3.o
