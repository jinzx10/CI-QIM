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
dc_adi64 = np.fromfile(dir64+'dc_adi.dat')
pes64 = np.fromfile(dir64+'E_adi.dat')

nx64 = len(xgrid64)
dc_adi64 = np.reshape(dc_adi64, (nx64,-1))
pes64 = np.reshape(pes64, (nx64,-1))


dir08 = datadir + '/0.0008/'
xgrid08 = np.fromfile(dir08+'xgrid.dat')
dc_adi08 = np.fromfile(dir08+'dc_adi.dat')
pes08 = np.fromfile(dir08+'E_adi.dat')

nx08 = len(xgrid08)
dc_adi08 = np.reshape(dc_adi08, (nx08,-1))
pes08 = np.reshape(pes08, (nx08,-1))

dir0025 = datadir + '/0.000025/'
xgrid0025 = np.fromfile(dir0025+'xgrid.dat')
dc_adi0025 = np.fromfile(dir0025+'dc_adi.dat')
pes0025 = np.fromfile(dir0025+'E_adi.dat')

nx0025 = len(xgrid0025)
dc_adi0025 = np.reshape(dc_adi0025, (nx0025,-1))
pes0025 = np.reshape(pes0025, (nx0025,-1))



J = [ i for i in range(1,30) ] 
dcmax64 = []
for i in range(1, 30):
    dcmax64.append(np.amax(np.abs(dc_adi64[:,i])))

dcmax08 = []
for i in range(1, 30):
    dcmax08.append(np.amax(np.abs(dc_adi08[:,i])))

dcmax0025 = []
for i in range(1, 30):
    dcmax0025.append(np.amax(np.abs(dc_adi0025[:,i])))

#print(dcmax0025)

############################################################
#                     plot
############################################################

fig = plt.figure(1)

sz_label=20
sz_legend=12
sz_tick=16
sz_title=20

# subplot 1
ax64 = fig.add_subplot(121)
ax64.plot(J, dcmax64,'-o')

ax64.set_xlabel(r"$J$", fontsize=sz_label)
ax64.set_ylabel(r"max$|d_{0J}|$", fontsize=sz_label)

ax64.set_xlim((0,30))
ax64.set_ylim((0,0.25))

ax64.tick_params(axis='x',direction='in', labelsize=sz_tick)
ax64.tick_params(axis='y',direction='in', labelsize=sz_tick)

ax64.set_title(r"$\Gamma=0.0064$", fontsize=sz_title)

#ax64.legend(fontsize=sz_legend)

# subplot 2
ax08 = fig.add_subplot(122)
ax08.plot(J, dcmax08,'-o')

ax08.set_xlabel(r"$J$", fontsize=sz_label)
ax08.set_ylabel(r"max$|d_{0J}|$", fontsize=sz_label)

ax08.set_xlim((0,30))
ax08.set_ylim((0,2))

ax08.tick_params(axis='x',direction='in', labelsize=sz_tick)
ax08.tick_params(axis='y',direction='in', labelsize=sz_tick)

ax08.set_title(r"$\Gamma=0.0008$", fontsize=sz_title)

fig.subplots_adjust(left=0.1,right=0.95,bottom=0.15,top=0.9,wspace=0.3)
fig.set_size_inches(9,4)
#plt.savefig('/home/zuxin/Dropbox/FSSH_rlx/tex/figs/dcmax.png', dpi=600)

plt.show()


