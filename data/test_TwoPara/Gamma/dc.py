import numpy as np
import matplotlib.pyplot as plt

#dirs = ['00001','00002','00004','00008','00016','00032','00064']
dirs = ['00016_800','00016_600','00016']

for dir_ in dirs:
    xgrid = np.loadtxt(dir_+'/xgrid.txt')
    dc = np.loadtxt(dir_+'/dc.txt')
    plt.plot(xgrid, np.abs(dc[1,:]))

plt.show()
