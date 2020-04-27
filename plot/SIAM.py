import numpy as np
import matplotlib.pyplot as plt 
import os

#rootdir = '/home/zuxin/job/CI-QIM/'
filedir=os.path.dirname(os.path.abspath(__file__))
datadir=filedir+'/../data/SIAM/test/'

num_figs = 3

xgrid = np.fromfile(datadir+'xgrid.dat')
n_mf= np.fromfile(datadir+'n_mf.dat')
n_cisnd= np.fromfile(datadir+'n_cisnd.dat')
dc_adi = np.fromfile(datadir+'dc_adi.dat')
gamma_rlx = np.fromfile(datadir+'Gamma_rlx.dat')

nx = len(xgrid)
n_cisnd = np.reshape(n_cisnd, (nx,-1))
dc_adi = np.reshape(dc_adi, (nx,-1))
gamma_rlx = np.reshape(gamma_rlx, (nx,-1))

plt.subplot(1,num_figs,1)
plt.plot(xgrid, n_mf) 
plt.plot(xgrid, n_cisnd[:,0]) 

plt.subplot(1,num_figs,2)
plt.plot(xgrid, dc_adi[:,1]) 

plt.subplot(1,num_figs,3)
plt.plot(xgrid, gamma_rlx[:,0])

plt.show()

