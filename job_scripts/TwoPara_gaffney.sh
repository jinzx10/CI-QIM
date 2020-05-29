#!/bin/bash

NUM_NODES=16
PROCS_PER_NODE=6
WALLTIME=24:00:00
QUEUE=standard
TOT_PROCS=$(bc -l <<< "${NUM_NODES}*${PROCS_PER_NODE}")

JOBROOT=${WORKDIR}/CI-QIM
LOG="run_TwoPara.log"
EXEC="run_TwoPara"
RAW_INPUT="TwoPara_raw.in"
INPUT="TwoPara.in"
JOB_SCRIPT="TwoPara.sh"
PBS_OUT="TwoPara.pbs.out"

mkdir -p ${JOBROOT}
mkdir -p ${JOBROOT}/bin ${JOBROOT}/tmp

cp ${HOME}/job/CI-QIM/bin/${EXEC} ${JOBROOT}/bin/${EXEC}
cp ${HOME}/job/CI-QIM/tmp/${RAW_INPUT} ${JOBROOT}/tmp/${RAW_INPUT}


DOS_BASE=( 1000    1000	   2000    4000    8000    16000   24000   24000   24000    24000)
HYBRID=(   0.0128  0.0064  0.0032  0.0016  0.0008  0.0004  0.0002  0.0001  0.00005  0.000025)
DOX_PEAK=( 20      30      40      60      100     140     200     300     450      700)
DOX_WIDTH=(5       4       3       2       1.4     1       0.7     0.5     0.3      0.2)


for i in {7..9}
do
	# data directory
	SAVEDIR=${JOBROOT}/data/TwoPara/hybrid_Gamma/${HYBRID[i]}
	mkdir -p ${SAVEDIR}
	rm -f ${SAVEDIR}/*.dat ${SAVEDIR}/*.txt ${SAVEDIR}/*.log

	# prepare input file
	cp ${JOBROOT}/tmp/${RAW_INPUT} ${SAVEDIR}/${INPUT}
	sed -i -e "s@SAVEDIR@${SAVEDIR}@" \
		-e "s/DOS_BASE/${DOS_BASE[i]}/" \
		-e "s/HYBRID/${HYBRID[i]}/" \
		-e "s/DOX_PEAK/${DOX_PEAK[i]}/" \
		-e "s/DOX_WIDTH/${DOX_WIDTH[i]}/" \
		${SAVEDIR}/${INPUT}

	cat > ${SAVEDIR}/${JOB_SCRIPT} <<-EOF
		#PBS -A AFOSR33712NSH
		#PBS -q ${QUEUE}
		#PBS -l select=${NUM_NODES}:ncpus=48:mpiprocs=${PROCS_PER_NODE}
		#PBS -l walltime=${WALLTIME}
		#PBS -N TwoPara
		#PBS -j oe
		#PBS -o ${SAVEDIR}/${PBS_OUT}
	
		mpiexec_mpt -np ${TOT_PROCS} ${JOBROOT}/bin/${EXEC} ${SAVEDIR}/${INPUT} >> ${SAVEDIR}/${LOG} 2>&1
	EOF

	# create log
	touch ${SAVEDIR}/${LOG}
	echo $(date) >> ${SAVEDIR}/${LOG}
	echo -e "\n-----------------job script start----------------\n" >> ${SAVEDIR}/${LOG}
	cat ${SAVEDIR}/${JOB_SCRIPT} >> ${SAVEDIR}/${LOG}
	echo -e "\n------------------job script end-----------------\n" >> ${SAVEDIR}/${LOG}

	echo -e "\n-----------------input file start----------------\n" >> ${SAVEDIR}/${LOG}
	cat ${SAVEDIR}/${INPUT} >> ${SAVEDIR}/${LOG}
	echo -e "\n------------------input file end-----------------\n" >> ${SAVEDIR}/${LOG}

	qsub ${SAVEDIR}/${JOB_SCRIPT}
done


