#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gs
import os
from matplotlib import rc


plt.rc('text', usetex=True)
plt.rc('font', family='serif')


############################################################
#                     Read Data
############################################################
filedir=os.path.dirname(os.path.abspath(__file__))
datadir=filedir+'/../data/TwoPara/hybrid_Gamma/'

folders = [
        '0.0064',
        '0.0008',
        '0.0001',
        '0.000025',
]

gamma_str = [ float(e) for e in folders ]
gamma = np.asarray(gamma_str)
gap = np.zeros(len(gamma_str))

for ig in range(0,len(gamma_str)):
    diri = datadir + folders[ig] + '/'
    xgrid = np.fromfile(diri + 'xgrid.dat')
    pes = np.fromfile(diri + 'E_adi.dat')
    nx = len(xgrid)
    pes = np.reshape(pes, (nx, -1))
    E01 = pes[:,1] - pes[:,0]
    gap[ig] = np.amin(E01)


plt.plot(gamma,gap,label='numerical')
plt.plot(gamma,gamma,label='Gamma')

plt.show()

