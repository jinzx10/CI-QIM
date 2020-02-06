import numpy as np
import matplotlib.pyplot as plt

#dirs = ['00001','00002','00004','00008','00016','00032','00064']
#dirs = ['00016_800','00016_600','00016']
dirs = []
for dir_ in dirs:
    xgrid = np.loadtxt(dir_+'/xgrid.txt')
    dc = np.loadtxt(dir_+'/dc.txt')
    plt.plot(xgrid, np.abs(dc[1,:]))

xgrid = np.loadtxt('0000025_1000/xgrid.txt')
dc = np.loadtxt('0000025_1000/dc01.txt')
plt.plot(xgrid, np.abs(dc))

plt.show()
