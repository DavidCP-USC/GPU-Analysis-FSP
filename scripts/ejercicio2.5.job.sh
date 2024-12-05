#!/bin/bash

#SBATCH -c 32
#SBATCH --mem 247G
#SBATCH --gres gpu:a100:1
#SBATCH -t 02:00:00
#SBATCH -J P3_2_5
#SBATCH -o ./outs/P3_2_5.o
#SBATCH -e ./outs/P3_2_5.e

rm ./data/P3_2_5.txt

nvcc ./code/vectorAdd_3.cu -o vectorAdd_5.o

start=3100000000
end=3480000000
step=10000000

for ((i=$start; i<=$end; i+=$step)); do
	srun vectorAdd_5.o $i >> ./data/P3_2_5.txt
done

rm vectorAdd_5.o
