#!/bin/bash

#SBATCH --chdir=/scratch/nas/3/angelm/ChampSim-ML

#SBATCH -c 4

#SBATCH --mem=8G

#SBATCH --nodes=1

#SBATCH --output=train-all-vulkansim-%j.log

memoryPerJob="16G"

#queue="small_gpu"
queue="large"

models_dir_name="results/MPMLP-tolerance-vulkansim/models"
logs_dir_name="results/MPMLP-tolerance-vulkansim/logs"
graphs_dir_name="results/MPMLP-tolerance-vulkansim/graphs"

train_trace_directory="VulkanSimTrain-sid-wid-tid"
test_trace_directory="VulkanSimTest-sid-wid-tid"

mkdir -p $models_dir_name
mkdir -p $logs_dir_name
mkdir -p $graphs_dir_name

for scene in BATH BUNNY CAR CHSNT CRNVL FOX FRST LANDS PARK PARTY REF ROBOT SHIP SPNZA SPRNG WKND;
do
	echo "Executing sbatch train_test_one_job.sh with arguments:" --output=${logs_dir_name}"/"${scene}"-%j.out" --error=${logs_dir_name}"/"${scene}"-error-%j.out" "--mem="$memoryPerJob "-c" 4 "-q" $queue ${train_trace_directory}"/"${scene}"-frame0.txt" ${test_trace_directory}"/"${scene}"-frame1.txt" ${models_dir_name}"/"${scene} ${graphs_dir_name}"/"${scene}
	sbatch --output=${logs_dir_name}"/"${scene}"-%j.out" --error=${logs_dir_name}"/"${scene}"-error-%j.out" --mem=$memoryPerJob -c 4 -q $queue train_test_one_job.sh ${train_trace_directory}"/"${scene}"-frame0.txt" ${test_trace_directory}"/"${scene}"-frame1.txt" ${models_dir_name}"/"${scene} ${graphs_dir_name}"/"${scene}
done