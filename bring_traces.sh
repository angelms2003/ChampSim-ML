#!/bin/bash

#SBATCH --chdir=/scratch/nas/3/angelm/ChampSim-ML

#SBATCH -c 4

#SBATCH --mem=8G

#SBATCH --nodes=1

#SBATCH --output=bring_traces-%j.log

for scene in BATH BUNNY CAR CHSNT CRNVL FOX FRST LANDS PARK PARTY REF ROBOT SHIP SPNZA SPRNG WKND;
do
    cp ../vulkan-sim/sim_run_11.1/${scene}_PT/*/RTX4060_rt_access_reader_1_L2/bin/L2_rt_memory_access_reader-frame0*.txt VulkanSimTrain/${scene}-frame0.txt
done

for scene in BATH BUNNY CAR CHSNT CRNVL FOX FRST LANDS PARK PARTY REF ROBOT SHIP SPNZA SPRNG WKND
do
    cp ../vulkan-sim/sim_run_11.1/${scene}_PT/*/RTX4060_rt_access_reader_1_L2/bin/L2_rt_memory_access_reader-frame1*.txt VulkanSimTest/${scene}-frame1.txt
done