#!/bin/bash

#SBATCH --chdir=/scratch/nas/3/angelm/ChampSim-ML

echo "Calling ./ml_prefetch_sim.py run" $1 "--prefetch" $2 "--results-dir" $3 "--no-base"

/usr/bin/time -v ./ml_prefetch_sim.py run $1 --prefetch $2 --results-dir $3 --no-base
