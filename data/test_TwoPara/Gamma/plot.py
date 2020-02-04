import numpy as np
import matplotlib.pyplot as plt

num_figs = 5
dir='00064/'

pes = np.loadtxt(dir+'./val_cis_sub.txt')
gamma = np.loadtxt(dir+'./Gamma.txt')
xgrid = np.loadtxt(dir+'./xgrid.txt')
V0 = np.loadtxt(dir+'./V0.txt')
Eg = np.loadtxt(dir+'./Eg.txt')
n_imp = np.loadtxt(dir+'./n_imp.txt')
force = np.loadtxt(dir+'./force.txt')
val_sub = np.loadtxt(dir+'./val_cis_sub.txt')
dc = np.loadtxt(dir+'./dc.txt')

plt.subplot(1,num_figs,1)
plt.plot(xgrid, Eg+V0)
plt.plot(xgrid, pes[0,:]+V0)
plt.plot(xgrid, pes[1,:]+V0)
plt.plot(xgrid, pes[2,:]+V0)

plt.subplot(1,num_figs,2)
plt.plot(xgrid, n_imp)

plt.subplot(1,num_figs,3)
plt.plot(xgrid, gamma[0,:])
plt.plot(xgrid, gamma[1,:])
plt.plot(xgrid, gamma[2,:])

plt.subplot(1,num_figs,4)
plt.plot(xgrid, force[0,:])
f_dx = -(Eg[0:-1]-Eg[1:]) / (xgrid[0:-1]-xgrid[1:])
plt.plot(xgrid[1:], f_dx)
plt.plot(xgrid, force[1,:])
f_dx = -(val_sub[0,0:-1]-val_sub[0,1:]) / (xgrid[0:-1]-xgrid[1:])
plt.plot(xgrid[1:], f_dx)

plt.subplot(1, num_figs, 5)
#plt.plot(xgrid, dc[0,:])
plt.plot(xgrid, np.abs(dc[1,:]))
#plt.plot(xgrid, dc[2,:])
#plt.plot(xgrid, dc[3,:])

plt.show()