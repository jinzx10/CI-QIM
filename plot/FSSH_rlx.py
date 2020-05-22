#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt 
import os
import sys

filedir=os.path.dirname(os.path.abspath(__file__))
datadir=filedir+'/../data/TwoPara/hybrid_Gamma/'+sys.argv[1]+'/'

t = np.fromfile(datadir+'t.dat')
x_t = np.fromfile(datadir+'x_t.dat')
xc = 8.00011

nt = len(t)
x_t = np.reshape(x_t, (nt, -1))

n_left = np.sum(x_t < xc, 1)
plt.plot(t, n_left)

plt.show()

