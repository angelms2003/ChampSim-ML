#!/bin/bash

#SBATCH --chdir=/scratch/nas/3/angelm/ChampSim-ML

echo "Calling ./ml_prefetch_sim.py train_and_test" $1 $2 "--model" $3 "--graph-name" $4

/usr/bin/time -v ./ml_prefetch_sim.py train_and_test $1 $2 --model $3 --graph-name $4
