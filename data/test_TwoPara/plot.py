import numpy as np
import matplotlib.pyplot as plt

pes = np.loadtxt('./val_cis_sub.txt')
gamma = np.loadtxt('./Gamma.txt')
xgrid = np.loadtxt('./xgrid.txt')
V0 = np.loadtxt('./V0.txt')
Eg = np.loadtxt('./Eg.txt')
n_imp = np.loadtxt('./n_imp.txt')

plt.subplot(1,3,1)
plt.plot(xgrid, Eg+V0)
plt.plot(xgrid, pes[0,:]+V0)
plt.plot(xgrid, pes[1,:]+V0)
plt.plot(xgrid, pes[2,:]+V0)

plt.subplot(1,3,2)
plt.plot(xgrid, n_imp)

plt.subplot(1,3,3)
plt.plot(xgrid, gamma[0,:])
plt.plot(xgrid, gamma[1,:])
plt.plot(xgrid, gamma[2,:])

plt.show()
