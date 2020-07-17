#!/bin/bash

TIMESTAMP=$(date +%Y%m%d%H%M%S)
JOBROOT=${WORKDIR}/CI-QIM
LOG="run_FSSH_rlx.${TIMESTAMP}.log"
EXEC="run_FSSH_rlx"
RAW_INPUT="FSSH_rlx_raw.in"
INPUT="FSSH_rlx.in"
JOB_SCRIPT="FSSH_rlx.sh"
PBS_OUT="FSSH_rlx.pbs.out"

mkdir -p ${JOBROOT}
mkdir -p ${JOBROOT}/bin ${JOBROOT}/tmp

cp ${HOME}/job/CI-QIM/bin/${EXEC} ${JOBROOT}/bin/${EXEC}
cp ${HOME}/job/CI-QIM/tmp/${RAW_INPUT} ${JOBROOT}/tmp/${RAW_INPUT}

NNODES=(   5          5          5          5          5          5          10         10         25         25)
PPN=(      48         48         48         48         48         48         48         48         48         48) 
WTIME=(    00:30:00   01:10:00   01:30:00   02:30:00   03:30:00   05:00:00   04:00:00   06:00:00   05:00:00   08:00:00 )
QQ=(       debug      standard   standard   standard   standard   standard   standard   standard   standard   standard )

HYBRID=(	 0.0128  0.0064  0.0032  0.0016  0.0008  0.0004 0.0002  0.0001  0.00005  0.000025)
T_MAX_LIST=( 1.5e5   3e5     5e5     1e6    1.4e6    2e6    3e6     5e6      1e7     1.5e7   )
DTC_LIST=(   30      30      20      20      15      15     10      10       10      10)
NUM_TRAJS=2400
SZ_ELEC=40
VELO_REV=0
VELO_RESCALE=1
HAS_RLX=1
FRIC_MODE=0

for i in {0..9}
do
	# data directory
	READDIR=${JOBROOT}/data/TwoPara/hybrid_Gamma/${HYBRID[i]}_sz100
	SAVEDIR=${JOBROOT}/data/TwoPara/hybrid_Gamma/${HYBRID[i]}_sz100/vr${VELO_REV}_rlx${HAS_RLX}_rsc${VELO_RESCALE}_sze${SZ_ELEC}
	if [ ${FRIC_MODE} == "-1" ]; then
		SAVEDIR+="_nofric"
	fi

	mkdir -p ${SAVEDIR}

	# prepare input file
	cp ${JOBROOT}/tmp/${RAW_INPUT} ${SAVEDIR}/${INPUT}
	sed -i -e "s@SAVEDIR@${SAVEDIR}@" \
		-e "s@READDIR@${READDIR}@" \
		-e "s/NUM_TRAJS/${NUM_TRAJS}/" \
		-e "s/T_MAX/${T_MAX_LIST[i]}/" \
		-e "s/DTC/${DTC_LIST[i]}/" \
		-e "s/SZ_ELEC/${SZ_ELEC}/" \
		-e "s/VELO_REV/${VELO_REV}/" \
		-e "s/VELO_RESCALE/${VELO_RESCALE}/" \
		-e "s/HAS_RLX/${HAS_RLX}/" \
		-e "s/FRIC_MODE/${FRIC_MODE}/" \
		${SAVEDIR}/FSSH_rlx.in

	QUEUE=${QQ[i]}
	NUM_NODES=${NNODES[i]}
	PROCS_PER_NODE=${PPN[i]}
	WALLTIME=${WTIME[i]}
	TOT_PROCS=$(bc -l <<< "${NUM_NODES}*${PROCS_PER_NODE}")

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


