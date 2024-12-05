#!/bin/bash

#SBATCH -c 32
#SBATCH --mem 247G
#SBATCH --gres gpu:a100:1
#SBATCH -t 02:00:00
#SBATCH -J P3_2_2
#SBATCH -o ./outs/P3_2_2.o
#SBATCH -e ./outs/P3_2_2.e

rm ./data/P3_2_2.txt

nvcc ./code/vectorAdd.cu -o vectorAdd_2.o

start=32
end=1024
step=32

for ((i=$start; i<=$end; i+=$step)); do
	srun vectorAdd_2.o 3480000000 $i >> ./data/P3_2_2.txt
done

rm vectorAdd_2.o
