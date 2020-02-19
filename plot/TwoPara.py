import numpy as np
import matplotlib.pyplot as plt 

num_figs = 5 

xgrid = np.loadtxt('./xgrid.txt')
xfine = np.loadtxt('./x_fine.txt')
E0 = np.loadtxt('./E0.txt')
E1 = np.loadtxt('./E1.txt')
F0 = np.loadtxt('./F0.txt')
F1 = np.loadtxt('./F1.txt')
Gamma = np.loadtxt('./Gamma.txt')
n_imp = np.loadtxt('./n_imp.txt')
dc01 = np.loadtxt('./dc01.txt')

E0_fine = np.loadtxt('./E0_fine.txt')
E1_fine = np.loadtxt('./E1_fine.txt')
F0_fine = np.loadtxt('./F0_fine.txt')
F1_fine = np.loadtxt('./F1_fine.txt')
Gamma_fine = np.loadtxt('./Gamma_fine.txt')
dc01_fine = np.loadtxt('./dc01_fine.txt')

plt.subplot(1,num_figs,1)
plt.plot(xgrid, E0) 
plt.plot(xgrid, E1) 
plt.plot(xfine, E0_fine, linestyle='dotted')
plt.plot(xfine, E1_fine, linestyle='dotted')

plt.subplot(1,num_figs,2)
plt.plot(xgrid, n_imp)

plt.subplot(1,num_figs,3)
plt.plot(xgrid, Gamma)
plt.plot(xfine, Gamma_fine, linestyle='dotted')

plt.subplot(1,num_figs,4)
plt.plot(xgrid, F0) 
plt.plot(xgrid, F1) 
f_dx_0 = -(E0[1:]-E0[0:-1]) / (xgrid[1:]-xgrid[0:-1])
f_dx_1 = -(E1[1:]-E1[0:-1]) / (xgrid[1:]-xgrid[0:-1])
plt.plot(xgrid[1:], f_dx_0)
plt.plot(xgrid[1:], f_dx_1)
plt.plot(xfine, F0_fine, linestyle='dotted')
plt.plot(xfine, F1_fine, linestyle='dotted')

plt.subplot(1, num_figs, 5)
plt.plot(xgrid, np.abs(dc01))
plt.plot(xfine, dc01_fine, linestyle='dotted')

plt.show()

