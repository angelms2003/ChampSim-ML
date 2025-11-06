#!/bin/bash

#SBATCH --chdir=/scratch/nas/3/angelm/ChampSim-ML

prefetcherName="MPMLP"

memoryPerJob="100G"

queue="big"

millionsOfTrainingInstructions="100"

generateDirectory="generate"$prefetcherName

mkdir $generateDirectory

mkdir logs

for loadTrace in $(find ML-DPC/LoadTraces/ -type f);
do
	if [[ "$loadTrace" == "ML-DPC/LoadTraces/gap/pr-3.txt.xz" ]]; then
		memoryPerJob="100G"
		queue="big"
	elif [[ "$loadTrace" == "ML-DPC/LoadTraces/gap/bc-12.txt.xz" ]]; then
		memoryPerJob="100G"
		queue="big"
	elif [[ "$loadTrace" == "ML-DPC/LoadTraces/gap/sssp-10.txt.xz" ]]; then
		memoryPerJob="100G"
		queue="big"
	elif [[ "$loadTrace" == "ML-DPC/LoadTraces/gap/cc-13.txt.xz" ]]; then
		memoryPerJob="100G"
		queue="big"
	elif [[ "$loadTrace" == "ML-DPC/LoadTraces/gap/bfs-14.txt.xz" ]]; then
		memoryPerJob="16G"
		queue="all"
	elif [[ "$loadTrace" == "ML-DPC/LoadTraces/spec17/621.wrf-s0.txt.xz" ]]; then
		memoryPerJob="8G"
		queue="all"
	elif [[ "$loadTrace" == "ML-DPC/LoadTraces/spec17/619.lbm-s3.txt.xz" ]]; then
		memoryPerJob="16G"
		queue="all"
	elif [[ "$loadTrace" == "ML-DPC/LoadTraces/spec17/623.xalancbmk-s2.txt.xz" ]]; then
		memoryPerJob="32G"
		queue="all"
	elif [[ "$loadTrace" == "ML-DPC/LoadTraces/spec17/605.mcf-s2.txt.xz" ]]; then
		memoryPerJob="100G"
		queue="big"
	elif [[ "$loadTrace" == "ML-DPC/LoadTraces/spec17/607.cactuBSSN-s2.txt.xz" ]]; then
		memoryPerJob="8G"
		queue="all"
	elif [[ "$loadTrace" == "ML-DPC/LoadTraces/spec06/429.mcf-s1.txt.xz" ]]; then
		memoryPerJob="100G"
		queue="big"
	elif [[ "$loadTrace" == "ML-DPC/LoadTraces/spec06/473.astar-s0.txt.xz" ]]; then
		memoryPerJob="8G"
		queue="all"
	elif [[ "$loadTrace" == "ML-DPC/LoadTraces/spec06/433.milc-s2.txt.xz" ]]; then
		memoryPerJob="16G"
		queue="all"
	elif [[ "$loadTrace" == "ML-DPC/LoadTraces/spec06/462.libquantum-s0.txt.xz" ]]; then
		memoryPerJob="100G"
		queue="big"
	elif [[ "$loadTrace" == "ML-DPC/LoadTraces/spec06/462.libquantum-s1.txt.xz" ]]; then
		memoryPerJob="100G"
		queue="big"
	fi

	echo "Executing sbatch train_one_job.sh with arguments:" --output="logs/"$(echo $loadTrace | cut -d'/' -f4 | awk -F'.xz' '{print $1}')"-%j.out" --error="logs/"$(echo $loadTrace | cut -d'/' -f4 | awk -F'.xz' '{print $1}')"-error-%j.out" "--mem="$memoryPerJob "-q" $queue $loadTrace ${generateDirectory}/$prefetcherName-$(echo $loadTrace | cut -d'/' -f4 | awk -F'.xz' '{print $1}') $millionsOfTrainingInstructions
	sbatch --output="logs/"$(echo $loadTrace | cut -d'/' -f4 | awk -F'.xz' '{print $1}')"-%j.out" --error="logs/"$(echo $loadTrace | cut -d'/' -f4 | awk -F'.xz' '{print $1}')"-error-%j.out" --mem=$memoryPerJob -q $queue train_one_job.sh $loadTrace ${generateDirectory}/$prefetcherName-$(echo $loadTrace | cut -d'/' -f4 | awk -F'.xz' '{print $1}') $millionsOfTrainingInstructions
done
