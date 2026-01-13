#!/bin/bash

#SBATCH --chdir=/scratch/nas/3/angelm/ChampSim-ML

#SBATCH -c 4

#SBATCH --mem=32G

#SBATCH --nodes=1

#SBATCH -q big

#SBATCH --output=run-notebook-%j.log

source .venv/bin/activate

jupyter nbconvert --execute --to notebook $1