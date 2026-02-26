#!/bin/bash

#SBATCH --chdir=/scratch/nas/3/angelm/ChampSim-ML

#SBATCH -c 4

#SBATCH --mem=8G

#SBATCH --nodes=1

#SBATCH --output=train-all-vulkansim-%j.log

memoryPerJob="16G"

#queue="small_gpu"
queue="large"

models_dir_name="models-delta-stateful-tolerance-20-with-lookahead-vulkansim"
logs_dir_name="logs-delta-stateful-tolerance-20-with-lookahead-vulkansim"
graphs_dir_name="graphs-delta-stateful-tolerance-20-with-lookahead-vulkansim"

train_trace_directory="VulkanSimTrain-warp-id"
test_trace_directory="VulkanSimTest-warp-id"

mkdir $models_dir_name
mkdir $logs_dir_name
mkdir $graphs_dir_name

for scene in BATH BUNNY CAR CHSNT CRNVL FOX FRST LANDS PARK PARTY REF ROBOT SHIP SPNZA SPRNG WKND;
do
	echo "Executing sbatch train_test_one_job.sh with arguments:" --output=${logs_dir_name}"/"${scene}"-%j.out" --error=${logs_dir_name}"/"${scene}"-error-%j.out" "--mem="$memoryPerJob "-c" 4 "-q" $queue ${train_trace_directory}"/"${scene}"-frame0.txt" ${test_trace_directory}"/"${scene}"-frame1.txt" ${models_dir_name}"/"${scene} ${graphs_dir_name}"/"${scene}
	sbatch --output=${logs_dir_name}"/"${scene}"-%j.out" --error=${logs_dir_name}"/"${scene}"-error-%j.out" --mem=$memoryPerJob -c 4 -q $queue train_test_one_job.sh ${train_trace_directory}"/"${scene}"-frame0.txt" ${test_trace_directory}"/"${scene}"-frame1.txt" ${models_dir_name}"/"${scene} ${graphs_dir_name}"/"${scene}
done