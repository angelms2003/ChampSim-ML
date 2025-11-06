#!/bin/bash

#SBATCH --chdir=/scratch/nas/3/angelm/ChampSim-ML

source .venv/bin/activate

pip install $1
