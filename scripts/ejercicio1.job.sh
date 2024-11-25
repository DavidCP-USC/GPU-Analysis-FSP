#!/bin/bash

rm ./data/P3_1_A100.txt
rm ./data/P3_1_T4.txt

nvcc ./code/devquery.cu -o dewquery.o

srun --gres=gpu:a100 -c 32 --mem=32G -t 1 dewquery.o >> ./data/P3_1_A100.txt
srun -p viz --gres=gpu:t4 --mem=8G -t 20 dewquery.o >> ./data/P3_1_T4.txt

rm dewquery.o