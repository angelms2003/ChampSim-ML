#!/bin/bash

#SBATCH --chdir=/scratch/nas/3/angelm/ChampSim-ML

#SBATCH -c 4

#SBATCH --mem=8G

#SBATCH --nodes=1

#SBATCH --output=optuna_train-all-vulkansim-%j.log

memoryPerJob="16G"

queue="huge"

execution_name="optuna-onlymiss-softmax-delta-stateful-tolerance-20-with-lookahead-global-tid-vulkansim"

models_dir_name="results/"${execution_name}"/models"
logs_dir_name="results/"${execution_name}"/logs"
graphs_dir_name="results/"${execution_name}"/graphs"
experiments_dir_name="results/"${execution_name}"/experiments"

train_trace_directory="VulkanSimTrain-sid-wid-tid-onlymiss"
test_trace_directory="VulkanSimTest-sid-wid-tid-onlymiss"

mkdir -p $models_dir_name
mkdir -p $logs_dir_name
mkdir -p $graphs_dir_name
mkdir -p $experiments_dir_name

for scene in BATH BUNNY CAR CHSNT CRNVL FOX FRST LANDS PARTY REF SHIP SPNZA SPRNG WKND PARK ROBOT;
do
	echo "Executing sbatch optuna_train_test_one_job.sh with arguments:" --output=${logs_dir_name}"/"${scene}"-%j.out" --error=${logs_dir_name}"/"${scene}"-error-%j.out" "--mem="$memoryPerJob "-c" 4 "-q" $queue $train_trace_directory"/"${scene}"-frame0.txt" $test_trace_directory"/"${scene}"-frame1.txt" ${models_dir_name}"/"${scene} ${graphs_dir_name}"/"${scene} ${experiments_dir_name}"/"${scene}
	sbatch --output=${logs_dir_name}"/"${scene}"-%j.out" --error=${logs_dir_name}"/"${scene}"-error-%j.out" --mem=$memoryPerJob -c 1 -q $queue optuna_train_test_one_job.sh $train_trace_directory"/"${scene}"-frame0.txt" $test_trace_directory"/"${scene}"-frame1.txt" ${models_dir_name}"/"${scene} ${graphs_dir_name}"/"${scene} ${experiments_dir_name}"/"${scene}
done