import numpy as np
import matplotlib.pyplot as plt

#dirs = ['00001','00002','00004','00008','00016','00032','00064']
dirs = ['00016_800','00016_600','00016']

for dir_ in dirs:
    xgrid = np.loadtxt(dir_+'/xgrid.txt')
    Gamma = np.loadtxt(dir_+'/Gamma.txt')
    plt.plot(xgrid, Gamma[0,:])

xgrid = np.loadtxt('./00016_1000/xgrid.txt')
gamma = np.loadtxt('./00016_1000/Gamma.txt')
plt.show()
