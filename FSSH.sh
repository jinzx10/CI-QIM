#!/bin/bash

#PBS -N run_TwoPara
#PBS -l nodes=2:ppn=48
#PBS -l walltime=2:00:00
#PBS -j oe
#PBS -o /home/jinzx10/job/CI-QIM/pbs.out/pbs.out

JOBROOT=/home/jinzx10/job/CI-QIM/
cd $JOBROOT

GAMMA_LIST=(    0.0128  0.0064  0.0032  0.0016  0.0008  0.0004 0.0002  0.0001  0.00005  0.000025)
MAX_TIME_LIST=(  1.5e5   3e5     5e5     1e6    1.4e6    2e6    3e6     5e6      1e7     2e7   )
DTC_LIST=(        30     30      20      20      15      15     10      10       5       5)
NUM_TRAJ=2000

for i in {0..5}
do
	GAMMA=${GAMMA_LIST[i]}
	READDIR=$JOBROOT/data/TwoPara/Gamma/$GAMMA
	SAVEDIR=$JOBROOT/data/FSSH/Gamma/$GAMMA
	MAX_TIME=${MAX_TIME_LIST[i]}
	DTC=${DTC_LIST[i]}
	mpirun -np 96 bin/run_FSSH $READDIR $SAVEDIR $NUM_TRAJ $MAX_TIME $DTC > $JOBROOT/pbs.out/FSSH.Gamma.$GAMMA.out
done
