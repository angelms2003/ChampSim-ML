#!/bin/bash

#SBATCH --chdir=/scratch/nas/3/angelm/ChampSim-ML

prefetcherName="MPMLP"

memoryPerJob="100G"



for champSimTrace in $(find ML-DPC/ChampSimTraces/ -type f);
do
	if [[ "$champSimTrace" == "ML-DPC/ChampSimTraces/spec17/602.gcc-s1.trace.xz" ]]; then
		memoryPerJob="100G"
	else
		memoryPerJob="100G"
	fi

	echo "Executing sbatch run_one_job.sh with arguments:" $memoryPerJob $champSimTrace generated/$prefetcherName-$(echo $champSimTrace | cut -d'/' -f4 | awk -F '.trace' '{print $1}').txt results-$prefetcherName
	sbatch --mem=$memoryPerJob run_one_job.sh $champSimTrace generated/$prefetcherName-$(echo $champSimTrace | cut -d'/' -f4 | awk -F '.trace' '{print $1}').txt results-$prefetcherName
done
