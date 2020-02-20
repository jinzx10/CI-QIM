import numpy as np
import matplotlib.pyplot as plt 

#rootdir = '/home/zuxin/job/CI-QIM/'
rootdir = '/home/jinzx10/job/CI-QIM/'
num_figs = 5 
datadir=rootdir+'data/TwoPara/Gamma/0.0016/'

xgrid = np.fromfile(datadir+'xgrid.dat')
E0 = np.fromfile(datadir+'E0.dat')
E1 = np.fromfile(datadir+'E1.dat')
F0 = np.fromfile(datadir+'F0.dat')
F1 = np.fromfile(datadir+'F1.dat')
Gamma = np.fromfile(datadir+'Gamma.dat')
n_imp = np.fromfile(datadir+'n_imp.dat')
dc01x = np.fromfile(datadir+'dc01x.dat')
dc01 = np.fromfile(datadir+'dc01.dat')

#xfine = np.fromfile('./x_fine.dat')
#E0_fine = np.fromfile('./E0_fine.dat')
#E1_fine = np.fromfile('./E1_fine.dat')
#F0_fine = np.fromfile('./F0_fine.dat')
#F1_fine = np.fromfile('./F1_fine.dat')
#Gamma_fine = np.fromfile('./Gamma_fine.dat')
#dc01_fine = np.fromfile('./dc01_fine.dat')

plt.subplot(1,num_figs,1)
plt.plot(xgrid, E0) 
plt.plot(xgrid, E1) 
#plt.plot(xfine, E0_fine, linestyle='dotted')
#plt.plot(xfine, E1_fine, linestyle='dotted')

plt.subplot(1,num_figs,2)
plt.plot(xgrid, n_imp)

plt.subplot(1,num_figs,3)
plt.plot(xgrid, Gamma)
#plt.plot(xfine, Gamma_fine, linestyle='dotted')

plt.subplot(1,num_figs,4)
plt.plot(xgrid, F0) 
plt.plot(xgrid, F1) 
f_dx_0 = -(E0[1:]-E0[0:-1]) / (xgrid[1:]-xgrid[0:-1])
f_dx_1 = -(E1[1:]-E1[0:-1]) / (xgrid[1:]-xgrid[0:-1])
plt.plot(xgrid[1:], f_dx_0)
plt.plot(xgrid[1:], f_dx_1)
#plt.plot(xfine, F0_fine, linestyle='dotted')
#plt.plot(xfine, F1_fine, linestyle='dotted')

plt.subplot(1, num_figs, 5)
plt.plot(xgrid, np.abs(dc01x))
plt.plot(xgrid, np.abs(dc01))
#plt.plot(xfine, dc01_fine, linestyle='dotted')

plt.show()

