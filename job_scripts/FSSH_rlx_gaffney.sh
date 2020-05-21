#!/bin/bash

#!/bin/bash

NUM_NODES=8
PROCS_PER_NODE=24
WALLTIME=0:30:00
QUEUE=debug
TOT_PROCS=$(bc -l <<< "${NUM_NODES}*${PROCS_PER_NODE}")

JOBROOT=${WORKDIR}/CI-QIM
LOG="run_FSSH_rlx.log"
EXEC="run_FSSH_rlx"
RAW_INPUT="FSSH_rlx_raw.in"
INPUT="FSSH_rlx.in"
JOB_SCRIPT="FSSH_rlx.sh"
PBS_OUT="FSSH_rlx.pbs.out"

mkdir -p ${JOBROOT}
mkdir -p ${JOBROOT}/bin ${JOBROOT}/tmp

cp ${HOME}/job/CI-QIM/bin/${EXEC} ${JOBROOT}/bin/${EXEC}
cp ${HOME}/job/CI-QIM/tmp/${RAW_INPUT} ${JOBROOT}/tmp/${RAW_INPUT}


HYBRID=(	 0.0128  0.0064  0.0032  0.0016  0.0008  0.0004 0.0002  0.0001  0.00005  0.000025)
T_MAX_LIST=( 1.5e5   3e5     5e5     1e6    1.4e6    2e6    3e6     5e6      1e7     2e7   )
DTC_LIST=(   30      30      20      20      15      15     10      10       5       5)
NUM_TRAJS=200
SZ_ELEC=30

for i in {0..2}
do
	# data directory
	SAVEDIR=${JOBROOT}/data/TwoPara/hybrid_Gamma/${HYBRID[i]}
	mkdir -p ${SAVEDIR}
	rm -f ${SAVEDIR}/${LOG}

	# prepare input file
	cp ${JOBROOT}/tmp/${RAW_INPUT} ${SAVEDIR}/${INPUT}
	sed -i -e "s@SAVEDIR@${SAVEDIR}@" \
		-e "s@READDIR@${SAVEDIR}@" \
		-e "s/NUM_TRAJS/${NUM_TRAJS}/" \
		-e "s/T_MAX/${T_MAX_LIST[i]}/" \
		-e "s/DTC/${DTC_LIST[i]}/" \
		-e "s/SZ_ELEC/${SZ_ELEC}/" \
		${SAVEDIR}/FSSH_rlx.in


	cat > ${SAVEDIR}/${JOB_SCRIPT} <<-EOF
		#PBS -A AFOSR33712NSH
		#PBS -q ${QUEUE}
		#PBS -l select=${NUM_NODES}:ncpus=48:mpiprocs=${PROCS_PER_NODE}
		#PBS -l walltime=${WALLTIME}
		#PBS -N FSSH_rlx
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


