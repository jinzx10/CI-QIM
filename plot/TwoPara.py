#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt 
import os
import sys

switch_fine = True

filedir=os.path.dirname(os.path.abspath(__file__))
datadir=filedir+'/../data/TwoPara/hybrid_Gamma/'+sys.argv[1]+'/'

num_fig_row = 2
num_fig_col = 3

xgrid = np.fromfile(datadir+'xgrid.dat')
n_imp = np.fromfile(datadir+'n_imp.dat')
dc_adi = np.fromfile(datadir+'dc_adi.dat')
gamma_rlx = np.fromfile(datadir+'Gamma_rlx.dat')
force = np.fromfile(datadir+'F_adi.dat')
pes = np.fromfile(datadir+'E_adi.dat')


nx = len(xgrid)

dc_adi = np.reshape(dc_adi, (nx,-1))
gamma_rlx = np.reshape(gamma_rlx, (nx,-1))
pes = np.reshape(pes, (nx,-1))
n_imp = np.reshape(n_imp, (nx,-1))
force = np.reshape(force, (nx,-1))

if switch_fine:
    x_fine = np.fromfile(datadir+'x_fine.dat')
    dc_fine  = np.fromfile(datadir+'dc_fine.dat')
    pes_fine = np.fromfile(datadir+'E_fine.dat')
    n_imp_fine = np.fromfile(datadir+'n_imp_fine.dat')
    force_fine = np.fromfile(datadir+'F_fine.dat')
    gamma_fine = np.fromfile(datadir+'Gamma_fine.dat')

    nx_fine = len(x_fine)
    dc_fine = np.reshape(dc_fine, (nx_fine,-1))
    gamma_fine = np.reshape(gamma_fine, (nx_fine,-1))
    force_fine = np.reshape(force_fine, (nx_fine,-1))
    pes_fine = np.reshape(pes_fine, (nx_fine,-1))
    n_imp_fine = np.reshape(n_imp_fine, (nx_fine,-1))

plt.subplot(num_fig_row, num_fig_col,1)
plt.plot(xgrid, n_imp[:,[0,1,2,3]]) 
if switch_fine:
    plt.plot(x_fine,n_imp_fine[:,[0,1,2,3]],linestyle=':',color='black')

plt.subplot(num_fig_row, num_fig_col,2)
plt.plot(xgrid, dc_adi[:,1]) 
plt.plot(xgrid, dc_adi[:,2]) 
plt.plot(xgrid, dc_adi[:,29]) 
if switch_fine:
    plt.plot(x_fine, dc_fine[:,1],linestyle=':')

plt.subplot(num_fig_row, num_fig_col,3)
plt.plot(xgrid, gamma_rlx[:,1])
if switch_fine:
    plt.plot(x_fine, gamma_fine[:,1],linestyle=':')

plt.subplot(num_fig_row, num_fig_col,4)
plt.plot(xgrid, pes[:,0:10:1])
if switch_fine:
    plt.plot(x_fine,pes_fine[:,[0,1]],linestyle=':')

plt.subplot(num_fig_row, num_fig_col,5)
plt.plot(xgrid, force[:,[0,1]])
f_fd = (pes[0:-1,[0,1]]-pes[1:,[0,1]])/np.reshape(xgrid[1:]-xgrid[0:-1],(-1,1))
plt.plot(xgrid[1:],f_fd, linestyle=':')
if switch_fine:
    plt.plot(x_fine, force_fine[:,[0,1]],linestyle='--')


plt.show()

