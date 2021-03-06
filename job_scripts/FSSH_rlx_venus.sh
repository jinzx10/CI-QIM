#!/bin/bash

#PBS -N run_FSSH_rlx
##PBS -q n8
#PBS -l nodes=5:ppn=48
#PBS -l walltime=168:00:00
#PBS -j oe
#PBS -o /home/jinzx10/job/CI-QIM/log/FSSH_rlx.out

JOBROOT=/home/jinzx10/job/CI-QIM/
cd $JOBROOT
TIMESTAMP=$(date +%Y%m%d%H%M%S)
LOG="run_fssh_rlx.${TIMESTAMP}.log"

HYBRID=(	 0.0128  0.0064  0.0032  0.0016  0.0008  0.0004 0.0002  0.0001  0.00005  0.000025)
T_MAX_LIST=( 1.5e5   3e5     5e5     1e6    1.4e6    2e6    3e6     5e6      1e7     1.5e7   )
DTC_LIST=(   30      30      20      20      15      15     10      10       10      10)
NUM_TRAJS=2400
SZ_ELEC=30
VELO_REV=0
HAS_RLX=0

for i in {5..9}
do
	READDIR=${JOBROOT}/data/TwoPara/hybrid_Gamma/${HYBRID[i]}
	SAVEDIR=${JOBROOT}/data/TwoPara/hybrid_Gamma/${HYBRID[i]}/vr${VELO_REV}_rlx${HAS_RLX}/
	mkdir -p ${SAVEDIR}

	cp ${JOBROOT}/tmp/FSSH_rlx_raw.in ${SAVEDIR}/FSSH_rlx.in

	sed -i -e "s@SAVEDIR@${SAVEDIR}@" \
		-e "s@READDIR@${READDIR}@" \
		-e "s/NUM_TRAJS/${NUM_TRAJS}/" \
		-e "s/T_MAX/${T_MAX_LIST[i]}/" \
		-e "s/DTC/${DTC_LIST[i]}/" \
		-e "s/VELO_REV/${VELO_REV}/" \
		-e "s/SZ_ELEC/${SZ_ELEC}/" \
		-e "s/HAS_RLX/${HAS_RLX}/" \
		${SAVEDIR}/FSSH_rlx.in

	touch ${SAVEDIR}/${LOG}
	echo $(date) >> ${SAVEDIR}/${LOG}

	mpirun -np 240 ${JOBROOT}/bin/run_FSSH_rlx ${SAVEDIR}/FSSH_rlx.in >> ${SAVEDIR}/${LOG} 2>&1
done


