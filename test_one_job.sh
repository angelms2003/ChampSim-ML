#!/bin/bash

#SBATCH --chdir=/scratch/nas/3/angelm/ChampSim-ML

#SBATCH -A gpu

#SBATCH -p gpu

#SBATCH --gres=gpu:1

echo "Calling ./ml_prefetch_sim.py test" $1 $2 "--model" $3 "--graph-name" $4

/usr/bin/time -v ./ml_prefetch_sim.py test $1 $2 --model $3 --graph-name $4
