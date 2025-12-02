#!/bin/bash

#SBATCH --chdir=/scratch/nas/3/angelm/ChampSim-ML

#SBATCH -c 4

#SBATCH --mem=8G

#SBATCH --nodes=1

#SBATCH --output=train-all-vulkansim-%j.log

prefetcherName="MPMLP"

memoryPerJob="100G"

queue="big"

# for scene in BATH BUNNY CAR CHSNT CRNVL FOX FRST LANDS PARK PARTY REF ROBOT SHIP SPNZA SPRNG WKND;
for scene in BATH BUNNY CAR CHSNT CRNVL FOX FRST LANDS PARTY REF ROBOT SHIP SPNZA SPRNG WKND;
do
	echo "Executing sbatch train_test_one_job.sh with arguments:" --output="logs/"${scene}"-%j.out" --error="logs/"${scene}"-error-%j.out" "--mem="$memoryPerJob "-c" 4 "-q" $queue "VulkanSimTrain/"${scene}"-frame0.txt" "VulkanSimTest/"${scene}"-frame1.txt" "models/"${scene}".pt" "graphs/"${scene}".png"
	sbatch --output="logs/"${scene}"-%j.out" --error="logs/"${scene}"-error-%j.out" --mem=$memoryPerJob -c 4 -q $queue train_test_one_job.sh "VulkanSimTrain/"${scene}"-frame0.txt" "VulkanSimTest/"${scene}"-frame1.txt" "models/"${scene}".pt" "graphs/"${scene}".png"
done