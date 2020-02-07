import numpy as np
import matplotlib.pyplot as plt

num_figs = 5

dir = './0000025/'
xgrid = np.loadtxt(dir+'./xgrid.txt')
E0 = np.loadtxt(dir+'./E0.txt')
E1 = np.loadtxt(dir+'./E1.txt')
F0 = np.loadtxt(dir+'./F0.txt')
F1 = np.loadtxt(dir+'./F1.txt')
Gamma = np.loadtxt(dir+'./Gamma.txt')
n_imp = np.loadtxt(dir+'./n_imp.txt')
dc01 = np.loadtxt(dir+'./dc01.txt')

plt.subplot(1,num_figs,1)
plt.plot(xgrid, E0)
plt.plot(xgrid, E1)

plt.subplot(1,num_figs,2)
plt.plot(xgrid, n_imp)

plt.subplot(1,num_figs,3)
plt.plot(xgrid, Gamma)

plt.subplot(1,num_figs,4)
plt.plot(xgrid, F0)
plt.plot(xgrid, F1)
f_dx_0 = -(E0[1:]-E0[0:-1]) / (xgrid[1:]-xgrid[0:-1])
f_dx_1 = -(E1[1:]-E1[0:-1]) / (xgrid[1:]-xgrid[0:-1])
plt.plot(xgrid[1:], f_dx_0)
plt.plot(xgrid[1:], f_dx_1)

plt.subplot(1, num_figs, 5)
plt.plot(xgrid, np.abs(dc01))

plt.show()
