#!/bin/bash

#PBS -N test_TwoPara
#PBS -l select=3:ncpus=32:mpiprocs=32
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -o /home/zuxin/job/CI-QIM/pbs.out

cd /home/zuxin/job/CI-QIM/

mpirun -np 96 ./bin/test_TwoPara
