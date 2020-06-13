#!/bin/bash

#PBS -N SIAM
#PBS -l nodes=4:ppn=24
#PBS -q batch
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /home/jinzx10/job/CI-QIM/log/SIAM.out

JOBROOT=/home/jinzx10/job/CI-QIM/
cd ${JOBROOT}

DOS_BASE=( 1000    1000	   2000    4000    8000    16000   24000   24000   24000    24000)
HYBRID=(   0.0128  0.0064  0.0032  0.0016  0.0008  0.0004  0.0002  0.0001  0.00005  0.000025)
DOX_PEAK=( 20      30      40      60      100     140     200     300     450      700)
DOX_WIDTH=(5       4       3       2       1.4     1       0.7     0.5     0.3      0.2)
SZ_SUB=(   50      50      100     200     400     800     1200    1200    1200     1200)
HUBBARD_U=0.06
OMEGA=0.0003

for i in {0..4}
do
	SAVEDIR=${JOBROOT}/data/SIAM/hybrid_Gamma/${HYBRID[i]}
	mkdir -p ${SAVEDIR}
	rm -f ${SAVEDIR}/*.dat ${SAVEDIR}/*.txt ${SAVEDIR}/*.log ${SAVEDIR}/*.in

	cp ${JOBROOT}/tmp/SIAM_raw.in ${SAVEDIR}/SIAM.in
	sed -i -e "s@SAVEDIR@${SAVEDIR}@" \
		-e "s/DOS_BASE/${DOS_BASE[i]}/" \
		-e "s/HYBRID/${HYBRID[i]}/" \
		-e "s/DOX_PEAK/${DOX_PEAK[i]}/" \
		-e "s/DOX_WIDTH/${DOX_WIDTH[i]}/" \
		-e "s/HUBBARD_U/${HUBBARD_U}/" \
		-e "s/OMEGA/${OMEGA}/" \
		-e "s/SZ_SUB/${SZ_SUB[i]}/" \
		${SAVEDIR}/SIAM.in

	touch ${SAVEDIR}/run_SIAM.log
	echo $(date) >> ${SAVEDIR}/run_SIAM.log

	mpirun -np 96 ${JOBROOT}/bin/run_SIAM ${SAVEDIR}/SIAM.in >> ${SAVEDIR}/run_SIAM.log 2>&1
done
