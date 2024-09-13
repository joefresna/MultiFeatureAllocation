#!/bin/bash
#Request 4 gigabytes of virtual (mem) and 2 of real (rmem) memory
#$ -l mem=4G -l rmem=2G
#$ -l h_rt=1:00:00
#$ -N multicat

if [ $# -le 2 ]; then
	#Load the Anaconda Python 3 Environment module for ICEBERG
	echo "Loading modules for iceberg"
	module load apps/python/anaconda3-2.5.0
else
	#Load the Anaconda Python 3 Environment module for SHARC
	echo "Loading modules for sharc"
	#module load apps/python/anaconda3-4.2.0
	source activate python35	 	 
fi


#Run the program
python $1 $2
