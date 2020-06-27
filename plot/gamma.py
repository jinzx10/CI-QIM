#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gs
import os
from matplotlib import rc

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


############################################################
#                     Read Data
############################################################

filedir=os.path.dirname(os.path.abspath(__file__))
datadir=filedir+'/../data/TwoPara/hybrid_Gamma/'

dir64 = datadir + '/0.0064/'
xgrid64 = np.fromfile(dir64+'xgrid.dat')
gamma64 = np.fromfile(dir64+'Gamma_rlx.dat')
nx64 = len(xgrid64)
gamma64 = np.reshape(gamma64, (nx64,-1))


dir08 = datadir + '/0.0008/'
xgrid08 = np.fromfile(dir08+'xgrid.dat')
gamma08 = np.fromfile(dir08+'Gamma_rlx.dat')
nx08 = len(xgrid08)
gamma08 = np.reshape(gamma08, (nx08,-1))


dir01 = datadir + '/0.0001/'
xgrid01 = np.fromfile(dir01+'xgrid.dat')
gamma01 = np.fromfile(dir01+'Gamma_rlx.dat')
nx01 = len(xgrid01)
gamma01 = np.reshape(gamma01, (nx01,-1))

dir0025 = datadir + '/0.000025/'
xgrid0025 = np.fromfile(dir0025+'xgrid.dat')
gamma0025 = np.fromfile(dir0025+'Gamma_rlx.dat')
nx0025 = len(xgrid0025)
gamma0025 = np.reshape(gamma0025, (nx0025,-1))


############################################################
#                     plot
############################################################

sz_label=20
sz_legend=12
sz_tick=16
sz_title=20

fig = plt.figure(1)

ax1 = fig.add_subplot(121)

ax1.plot(xgrid64, gamma64[:,1], label=r"$\Gamma=6.4\times 10^{-3}$")
ax1.plot(xgrid08, gamma08[:,1], label=r"$\Gamma=8\times 10^{-4}$")
ax1.plot(xgrid01, gamma01[:,1], label=r"$\Gamma=1\times 10^{-4}$")
ax1.plot(xgrid0025, gamma0025[:,1], label=r"$\Gamma=2.5\times 10^{-5}$")

ax1.set_xlim((-10,30))
ax1.set_yscale('log')

ax1.set_xlabel(r"$x$", fontsize=sz_label)
ax1.set_ylabel(r"$\tilde{\Gamma}_1$", fontsize=sz_label)
ax1.tick_params(axis='x',direction='in', labelsize=sz_tick)
ax1.tick_params(axis='y',direction='in', labelsize=sz_tick)
ax1.tick_params(which='minor',axis='y',direction='in', labelsize=sz_tick)

ax1.legend(fontsize=sz_legend, ncol=2, bbox_to_anchor=(0.3,0.7,0.7,0.3))
ax1.text(0.95, 0.15, '(a)', transform=ax1.transAxes, fontsize=sz_label, va='top', ha='right')


ax2 = fig.add_subplot(122)

ax2.plot(xgrid08, gamma08[:,1], label=r"$J=1$")
ax2.plot(xgrid08, gamma08[:,2], label=r"$J=2$")
ax2.plot(xgrid08, gamma08[:,3], label=r"$J=5$")
ax2.plot(xgrid08, gamma08[:,10], label=r"$J=10$")

ax2.set_xlim((6,10))
ax2.set_ylim((0,0.002))

ax2.set_xlabel(r"$x$", fontsize=sz_label)
ax2.set_ylabel(r"$\tilde{\Gamma}_J$", fontsize=sz_label)
ax2.tick_params(axis='x',direction='in', labelsize=sz_tick)
ax2.tick_params(axis='y',direction='in', labelsize=sz_tick)
ax2.tick_params(which='minor',axis='y',direction='in', labelsize=sz_tick)

ax2.legend(fontsize=sz_legend, ncol=1)
ax2.set_title(r"$\Gamma=0.0008$", fontsize=sz_title)
ax2.text(0.95, 0.15, '(b)', transform=ax2.transAxes, fontsize=sz_label, va='top', ha='right')

fig.subplots_adjust(left=0.1,right=0.96,bottom=0.15,top=0.9,wspace=0.36)
fig.set_size_inches(10,4)

plt.savefig('/home/zuxin/Dropbox/Anderson_Holstein/article2/tex/figs/gamma.png', dpi=600)
plt.show()




