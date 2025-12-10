#!/bin/bash

#SBATCH --chdir=/scratch/nas/3/angelm/ChampSim-ML

#SBATCH -A gpu

#SBATCH -p gpu

#SBATCH --gres=gpu:1

echo "Calling ./ml_prefetch_sim.py train" $1 "--generate" $2 "--num-prefetch-warmup-instructions" $3

/usr/bin/time -v ./ml_prefetch_sim.py train $1 --generate $2 --num-prefetch-warmup-instructions $3
