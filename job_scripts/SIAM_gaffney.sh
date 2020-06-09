#!/bin/bash

JOBROOT=${WORKDIR}/CI-QIM
LOG="run_SIAM.log"
EXEC="run_SIAM"
RAW_INPUT="SIAM_raw.in"
INPUT="SIAM.in"
JOB_SCRIPT="SIAM.sh"
PBS_OUT="SIAM.pbs.out"

mkdir -p ${JOBROOT}
mkdir -p ${JOBROOT}/bin ${JOBROOT}/tmp

cp ${HOME}/job/CI-QIM/bin/${EXEC} ${JOBROOT}/bin/${EXEC}
cp ${HOME}/job/CI-QIM/tmp/${RAW_INPUT} ${JOBROOT}/tmp/${RAW_INPUT}

NNODES=(   4          4          8          16          4         24         24         24         24         24)
PPN=(      24         24         24         12         12         4          4          4          4          4)
WTIME=(    00:10:00   00:10:00   00:20:00   00:30:00   00:30:00   08:00:00   24:00:00   24:00:00   24:00:00   24:00:00 )
QQ=(       debug      debug      debug      debug      debug      standard   standard   standard   standard   standard )

DOS_BASE=( 1000       1000       2000       4000       8000       16000      24000      24000      24000      24000)
HYBRID=(   0.0128     0.0064     0.0032     0.0016     0.0008     0.0004     0.0002     0.0001     0.00005    0.000025)
DOX_PEAK=( 20         30         40         60         100        140        200        300        450        700)
DOX_WIDTH=(5          4          3          2          1.4        1          0.7        0.5        0.3        0.2)
SZ_SUB=(   50         50         100        200        400        800        1200       1200       1200       1200)
HUBBARD_U=0.06
OMEGA=0.0003

for i in {4..4}
do
	# data directory
	SAVEDIR=${JOBROOT}/data/SIAM/hybrid_Gamma/${HYBRID[i]}_new
	mkdir -p ${SAVEDIR}
	rm -f ${SAVEDIR}/*.dat ${SAVEDIR}/*.txt ${SAVEDIR}/*.log

	# prepare input file
	cp ${JOBROOT}/tmp/${RAW_INPUT} ${SAVEDIR}/${INPUT}
	sed -i -e "s@SAVEDIR@${SAVEDIR}@" \
		-e "s/DOS_BASE/${DOS_BASE[i]}/" \
		-e "s/HYBRID/${HYBRID[i]}/" \
		-e "s/DOX_PEAK/${DOX_PEAK[i]}/" \
		-e "s/DOX_WIDTH/${DOX_WIDTH[i]}/" \
		-e "s/HUBBARD_U/${HUBBARD_U}/" \
		-e "s/OMEGA/${OMEGA}/" \
		-e "s/SZ_SUB/${SZ_SUB[i]}/" \
		${SAVEDIR}/${INPUT}

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
		#PBS -N SIAM
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


