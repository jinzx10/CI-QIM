import numpy as np
import matplotlib.pyplot as plt

#dirs = ['00001','00002','00004','00008','00016','00032','00064']
dirs = ['00016_800','00016_600','00016']
for dir_ in dirs:
    xgrid = np.loadtxt(dir_+'/xgrid.txt')
    Eg = np.loadtxt(dir_+'/Eg.txt')
    V0 = np.loadtxt(dir_+'/V0.txt')
    val_cis_sub = np.loadtxt(dir_+'/val_cis_sub.txt')
    pes0 = V0 + Eg
    pes1 = V0 + val_cis_sub[0,:]
    plt.plot(xgrid, pes0-pes0.min())
    plt.plot(xgrid, pes1-pes0.min())

xgrid = np.loadtxt('./00016_1000/xgrid.txt')
E0 = np.loadtxt('./00016_1000/E0.txt')
E1 = np.loadtxt('./00016_1000/E1.txt')
plt.plot(xgrid, E0 - E0.min())
plt.plot(xgrid, E1 - E0.min())
plt.show()
