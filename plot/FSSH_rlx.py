#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt 
import os
import sys
from scipy.optimize import curve_fit

def func(x,a,b,c):
    return a*np.exp(-b*x) + c

filedir=os.path.dirname(os.path.abspath(__file__))
datadir=filedir+'/../data/TwoPara/hybrid_Gamma/'+sys.argv[1]+'/'

nfit=500

t = np.fromfile(datadir+'t.dat')
x_t = np.fromfile(datadir+'x_t.dat')
xc = 8.00011

nt = len(t)

stride=nt//nfit

x_t = np.reshape(x_t, (-1, nt))
n_traj = np.size(x_t, 0)

n_left = np.sum((x_t < xc), 0) / n_traj;

t=t[::stride]
n_left=n_left[::stride]

idx=np.where( (n_left < 0.9) & (n_left > 0.1) )
t=t[idx]
n_left=n_left[idx]


#plt.plot(t[0::stride], n_left[0::stride])
#plt.show()


p = np.polyfit(t, np.log(n_left), 1)
print(p)

[coef,cov] = curve_fit(func, t, n_left, (1,1e-6,0))
print(coef)



