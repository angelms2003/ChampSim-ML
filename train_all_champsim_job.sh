#!/bin/bash

#SBATCH --chdir=/scratch/nas/3/angelm/ChampSim-ML

#SBATCH -c 4

#SBATCH --mem=8G

#SBATCH --nodes=1

#SBATCH --output=train-all-champsim-%j.log

memoryPerJob="16G"

queue="small_gpu"

models_dir_name="models-lstm-tolerance-20-with-lookahead-champsim"
logs_dir_name="logs-lstm-tolerance-20-with-lookahead-champsim"
graphs_dir_name="graphs-lstm-tolerance-20-with-lookahead-champsim"

mkdir $models_dir_name
mkdir $logs_dir_name
mkdir $graphs_dir_name

for scene in 429.mcf-s1 433.milc-s2 462.libquantum-s0 462.libquantum-s1 473.astar-s0 605.mcf-s2 607.cactuBSSN-s2 619.lbm-s3 621.wrf-s0 623.xalancbmk-s2 bc-12 bfs-14 cc-13 pr-3 sssp-10;
do
	echo "Executing sbatch train_test_one_job.sh with arguments:" --output=${logs_dir_name}"/"${scene}"-%j.out" --error=${logs_dir_name}"/"${scene}"-error-%j.out" "--mem="$memoryPerJob "-c" 4 "-q" $queue "ChampSimTrain/"${scene}".txt.xz" "ChampSimTest/"${scene}".txt.xz" ${models_dir_name}"/"${scene} ${graphs_dir_name}"/"${scene}
	sbatch --output=${logs_dir_name}"/"${scene}"-%j.out" --error=${logs_dir_name}"/"${scene}"-error-%j.out" --mem=$memoryPerJob -c 4 -q $queue train_test_one_job.sh "ChampSimTrain/"${scene}".txt.xz" "ChampSimTest/"${scene}".txt.xz" ${models_dir_name}"/"${scene} ${graphs_dir_name}"/"${scene}
done