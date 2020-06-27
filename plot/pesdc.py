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

mass = 2000
omega = 0.0002
dE=-0.0038
x0=0
x1=20.6097
diab64_0 = 0.5*mass*omega*omega*(xgrid64-x0)**2
diab64_1 = 0.5*mass*omega*omega*(xgrid64-x1)**2+dE
diab08_0 = 0.5*mass*omega*omega*(xgrid08-x0)**2
diab08_1 = 0.5*mass*omega*omega*(xgrid08-x1)**2+dE

############################################################
#                     plot
############################################################

# subplot 1
sz_label=20
sz_legend=12
sz_tick=16
sz_title=20

ax64pes = plt.subplot(1,2,1)
pescolor='tab:blue'
diabcolor='black'
ax64pes.plot(xgrid64, pes64[:,0:11:1] + 20.1, color=pescolor)
ax64pes.plot(xgrid64, diab64_0, color=diabcolor, linestyle='--')
ax64pes.plot(xgrid64, diab64_1, color=diabcolor, linestyle='--')
ax64pes.set_xlabel(r"$x$", fontsize=sz_label)
ax64pes.set_ylabel('PES', color=pescolor, fontsize=sz_label)
ax64pes.set_title(r"$\Gamma=0.0064$", fontsize=sz_title)
ax64pes.tick_params(axis='x', direction='in', labelsize=sz_tick)
ax64pes.tick_params(axis='y', labelcolor=pescolor, direction='in', labelsize=sz_tick)
ax64pes.set_xlim((-10,30))
ax64pes.set_ylim((-0.01, 0.02))
ax64pes.text(0.95, 0.95, '(a)', transform=ax64pes.transAxes, fontsize=sz_label, va='top', ha='right')

ax64dc = ax64pes.twinx()
dccolor='tab:red'
ax64dc.plot(xgrid64, -dc_adi64[:,1], color=dccolor, linestyle='-.', label='$J=1$')
ax64dc.plot(xgrid64, -dc_adi64[:,10], color=dccolor, linestyle=':', label='$J=10$')
ax64dc.set_ylabel('derivative coupling', color=dccolor, fontsize=sz_label)
ax64dc.tick_params(axis='y', labelcolor=dccolor, direction='in', labelsize=sz_tick)
ax64dc.set_ylim((-0.25, 2))
ax64dc.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
ax64dc.legend(fontsize=sz_legend, bbox_to_anchor=(0.8, 0.4, 0.2, 0.2))


# subplot 2
ax08pes = plt.subplot(1,2,2)
pescolor='tab:blue'
ax08pes.plot(xgrid08, pes08[:,0:11:1] + 160.1, color=pescolor)
ax08pes.plot(xgrid08, diab08_0, color=diabcolor, linestyle='--')
ax08pes.plot(xgrid08, diab08_1, color=diabcolor, linestyle='--')
ax08pes.set_xlabel(r"$x$", fontsize=sz_label)
ax08pes.set_ylabel('PES', color=pescolor, fontsize=sz_label)
ax08pes.set_title(r"$\Gamma=0.0008$", fontsize=sz_title)
ax08pes.tick_params(axis='x', direction='in', labelsize=sz_tick)
ax08pes.tick_params(axis='y', labelcolor=pescolor, direction='in', labelsize=sz_tick)
ax08pes.set_xlim((-10,30))
ax08pes.set_ylim((-0.01, 0.02))
ax08pes.text(0.95, 0.95, '(b)', transform=ax08pes.transAxes, fontsize=sz_label, va='top', ha='right')

ax08dc = ax08pes.twinx()
dccolor='tab:red'
ax08dc.plot(xgrid08, dc_adi08[:,1], color=dccolor, linestyle='-.', label='$J=1$')
ax08dc.plot(xgrid08, dc_adi08[:,10], color=dccolor, linestyle=':', label='$J=10$')
ax08dc.set_ylabel('derivative coupling', color=dccolor, fontsize=sz_label)
ax08dc.tick_params(axis='y', labelcolor=dccolor, direction='in', labelsize=sz_tick)
ax08dc.set_ylim((-0.25, 2))
ax08dc.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
ax08dc.legend(fontsize=sz_legend, loc='best', bbox_to_anchor=(0.8, 0.4, 0.2, 0.2))


fig = plt.gcf()
#fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.subplots_adjust(left=0.12,right=0.9,bottom=0.15,top=0.9,wspace=0.7)
fig.set_size_inches(10,4)

plt.savefig('/home/zuxin/Dropbox/Anderson_Holstein/article2/tex/figs/pesdc.png', dpi=600)
plt.show()


