#!/bin/bash

#PROJECT_HOME="/Users/joefresna/MultiCategorisation"
PROJECT_HOME="/home/ac1ar/MultiCategorisation"
CONF_DIR="${PROJECT_HOME}/conf_cluster"
mkdir -p ${CONF_DIR}
TEMPLATE_SETTINGS="${PROJECT_HOME}/conf/MultiCat.template.config"
PYPATH="${PROJECT_HOME}/src/"
EXEC_FILE="${PROJECT_HOME}/src/MultiCat/CategorisationProcess.py"
OUTPUT_DATA_DIR="${PROJECT_HOME}/data/neighs/"

SEED=22890
NUM_EXP=1000
ITERS=100

NUM_FEATURES_LIST=$(seq 2 1 10)
#NUM_FEATURES_LIST=(6)
ALLOC_TYPE_LIST=('random' 'split' 'on-noise' 'on-noise-quad' 'incremental-pre' 'incremental-post')

NEIGHS_LIST=( 3 5 7 11 )
#NEIGHS_LIST=( 3 )
NUM_AGENTS_LIST=(200 500 1000)

COUNT=0

for NUM_AGENTS in ${NUM_AGENTS_LIST[*]}
do
	for NEIGHS in ${NEIGHS_LIST[*]}
	do
		for NUM_FEATURES in ${NUM_FEATURES_LIST[*]}
		do
			for ALLOC_TYPE in ${ALLOC_TYPE_LIST[*]}
			do
				ACCURACIES=""
				for i in `awk 'BEGIN{ for(i=1;i<='${NUM_FEATURES}';i++) print i}'`
				do
					VAL=$(echo ${NUM_FEATURES} ${i} | awk '{printf "%1.5f\n", 0.5 + ( $2 * 0.5 / ( $1 +1 ) ) }')
					#VAL=$i
					ACCURACIES="${ACCURACIES} ${VAL},"
				done
				ACCURACIES=${ACCURACIES: :${#ACCURACIES}-1}
				
				#a-${ACCURACIES}_
				JOB_PARAM="agents-${NUM_AGENTS}_neighs-${NEIGHS}_features-${NUM_FEATURES}_method-${ALLOC_TYPE}"
				OUT_FILE="${OUTPUT_DATA_DIR}out_${JOB_PARAM}.txt"	
				
				CONF_FILE="${CONF_DIR}/muc_${JOB_PARAM}.config"
					
				sed -e "s|RND_SEED|${SEED}|" \
					-e "s|NUM_EXP|${NUM_EXP}|" \
					-e "s|ITERS|${ITERS}|" \
					-e "s|NUM_FEATURES|${NUM_FEATURES}|" \
					-e "s|ACCURACIES|${ACCURACIES}|" \
					-e "s|ALLOC_TYPE|${ALLOC_TYPE}|" \
					-e "s|NEIGHS|${NEIGHS}|" \
					-e "s|NUM_AGENTS|${NUM_AGENTS}|" \
					-e "s|OUT_FILE|${OUT_FILE}|" \
						${TEMPLATE_SETTINGS} > ${CONF_FILE}
						
				export PYTHONPATH=${PYPATH}
				#COMMAND="python3 ${EXEC_FILE} ${CONF_FILE}"
				#COMMAND="./run_job.sh ${EXEC_FILE} ${CONF_FILE}"
				COMMAND="qsub run_job.sh ${EXEC_FILE} ${CONF_FILE}"
				#${COMMAND}
				while ! ${COMMAND}
				do
					sleep 2
				done
				COUNT=$((COUNT + 1))
			done
		done
	done
done

echo "Submitted " ${COUNT} " jobs"
