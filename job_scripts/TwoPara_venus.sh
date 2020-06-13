#!/bin/bash

#PBS -N run_TwoPara
#PBS -l nodes=5:ppn=12
#PBS -l pmem=4gb
#PBS -q batch
#PBS -l walltime=168:00:00
#PBS -j oe
#PBS -o /home/jinzx10/job/CI-QIM/log/TwoPara.out

JOBROOT=/home/jinzx10/job/CI-QIM/
LOG="run_TwoPara.log"
cd ${JOBROOT}

DOS_BASE=( 1000    1000	   2000    4000    8000    16000   16000   16000   16000    16000)
HYBRID=(   0.0128  0.0064  0.0032  0.0016  0.0008  0.0004  0.0002  0.0001  0.00005  0.000025)
DOX_PEAK=( 20      30      40      60      100     140     200     300     450      700)
DOX_WIDTH=(5       4       3       2       1.4     1       0.7     0.5     0.3      0.2)

for i in {5..9}
do
	SAVEDIR=${JOBROOT}/data/TwoPara/hybrid_Gamma/${HYBRID[i]}_more
	mkdir -p ${SAVEDIR}
	rm -f ${SAVEDIR}/*.dat ${SAVEDIR}/*.txt ${SAVEDIR}/*.log

	cp ${JOBROOT}/tmp/TwoPara_raw.in ${SAVEDIR}/TwoPara.in
	sed -i -e "s@SAVEDIR@${SAVEDIR}@" \
		-e "s/DOS_BASE/${DOS_BASE[i]}/" \
		-e "s/HYBRID/${HYBRID[i]}/" \
		-e "s/DOX_PEAK/${DOX_PEAK[i]}/" \
		-e "s/DOX_WIDTH/${DOX_WIDTH[i]}/" \
		${SAVEDIR}/TwoPara.in

	touch ${SAVEDIR}/${LOG}
	echo $(date) >> ${SAVEDIR}/${LOG}

	mpirun -np 60 ${JOBROOT}/bin/run_TwoPara ${SAVEDIR}/TwoPara.in >> ${SAVEDIR}/${LOG} 2>&1
done
