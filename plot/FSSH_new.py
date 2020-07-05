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
n_t = np.fromfile(datadir+'n_t.dat')

nt = len(t)

stride=nt//nfit

n_t = np.reshape(n_t, (-1, nt))
n_traj = np.size(n_t, 0)

n_left = 1.0 - np.sum(n_t, 0) / n_traj;

t=t[::stride]
n_left=n_left[::stride]

idx=np.where( (n_left < 0.9) & (n_left > 0.1) )
t=t[idx]
n_left=n_left[idx]

plt.plot(t,n_left)
plt.show()

#plt.plot(t[0::stride], n_left[0::stride])
#plt.show()


p = np.polyfit(t, np.log(n_left), 1)
print('Aexp(-kt)   = ', -p[0])

[coef,cov] = curve_fit(func, t, n_left, (1,1e-6,0))
print('Aexp(-kt)+B = ', coef[1])



