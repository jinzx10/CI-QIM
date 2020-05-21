#!/bin/bash

#PBS -N run_FSSH_rlx
#PBS -l nodes=1:ppn=48
#PBS -l walltime=2:00:00
#PBS -j oe
#PBS -o /home/jinzx10/job/CI-QIM/log/FSSH_rlx.out

JOBROOT=/home/jinzx10/job/CI-QIM/
cd $JOBROOT

HYBRID=(	 0.0128  0.0064  0.0032  0.0016  0.0008  0.0004 0.0002  0.0001  0.00005  0.000025)
T_MAX_LIST=( 1.5e5   3e5     5e5     1e6    1.4e6    2e6    3e6     5e6      1e7     2e7   )
DTC_LIST=(   30      30      20      20      15      15     10      10       5       5)
NUM_TRAJS=24
LOG="run_fssh_rlx.log"

for i in {0..0}
do
	SAVEDIR=${JOBROOT}/data/TwoPara/hybrid_Gamma/${HYBRID[i]}
	mkdir -p ${SAVEDIR}
	rm -f ${SAVEDIR}/${LOG}

	cp ${JOBROOT}/tmp/FSSH_rlx_raw.in ${SAVEDIR}/FSSH_rlx.in

	sed -i -e "s@SAVEDIR@${SAVEDIR}@" \
		-e "s@READDIR@${SAVEDIR}@" \
		-e "s/NUM_TRAJS/${NUM_TRAJS}/" \
		-e "s/T_MAX/${T_MAX_LIST[i]}/" \
		-e "s/DTC/${DTC_LIST[i]}/" \
		${SAVEDIR}/FSSH_rlx.in

	touch ${SAVEDIR}/${LOG}
	echo $(date) >> ${SAVEDIR}/${LOG}

	mpirun -np 24 ${JOBROOT}/bin/run_FSSH_rlx ${SAVEDIR}/FSSH_rlx.in >> ${SAVEDIR}/${LOG} 2>&1
done


