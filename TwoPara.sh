#!/bin/bash

#PBS -N run_TwoPara
#PBS -l nodes=3:ppn=48
#PBS -l walltime=2:00:00
#PBS -j oe
#PBS -o /home/jinzx10/job/CI-QIM/pbs.out/pbs.out

JOBROOT=/home/jinzx10/job/CI-QIM/
cd $JOBROOT

DOS_BASE=5
GAMMA_LIST=(    0.0128  0.0064  0.0032  0.0016  0.0008  0.0004 0.0002  0.0001  0.00005  0.000025)
NUM_BATH_LIST=(   200    200     250     500     1000    2000   4000   8000     16000    32000)
DOS_PEAK_LIST=(   20     30      40      60      100     140    200    300      450      700)
DOS_WIDTH_LIST=(  5      4       3       2       1.4     1      0.7    0.5      0.3      0.2)

for i in {0..5}
do
	GAMMA=${GAMMA_LIST[i]}
	NUM_BATH=${NUM_BATH_LIST[i]}
	DOS_PEAK=${DOS_PEAK_LIST[i]}
	DOS_WIDTH=${DOS_WIDTH_LIST[i]}
	SAVEDIR=$JOBROOT/data/TwoPara/Gamma/$GAMMA
	echo $SAVEDIR $NUM_BATH $GAMMA $DOS_BASE $DOS_PEAK $DOS_WIDTH
	mpirun -np 144 bin/run_TwoPara $SAVEDIR $NUM_BATH $GAMMA $DOS_BASE $DOS_PEAK $DOS_WIDTH > $JOBROOT/pbs.out/TwoPara.Gamma.$GAMMA.out
done
