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

folders = [
        '0.0128',
        '0.0064',
        '0.0032',
        '0.0016',
        '0.0008',
        '0.0004',
        '0.0002',
        '0.0001',
        '0.00005',
        '0.000025',
]

r = [
    0.297623937290222, 
    0.391486478005056,
    0.477815977278981,
    0.557408259168905,
    0.630992080875044,
    0.699222044196402,
    0.762678883635222,
    0.821873867630406,
    0.877255161926642,
    0.929214838575803,
]

r2 = [ 
    0.536456871338239,
    0.615308283222295, 
    0.687644714396756,
    0.754330997689560,
    0.816085469954799,
    0.873509268046410,
    0.927108783974088,
    0.977313232208404,
    1.024488552899759,
    1.068948500984437,
]

gap = [
        1.08786553e-02,
        6.02852273e-03, 
        3.26299749e-03, 
        1.74323612e-03,
        9.22066829e-04, 
        4.83982984e-04, 
        2.53057302e-04, 
        1.32797295e-04,
        7.08120317e-05, 
        3.90797301e-05,
        ]
gamma_str = [ float(e) for e in folders ]
gamma = np.asarray(gamma_str)


############################################################
#                     plot
############################################################

sz_label=20
sz_legend=12
sz_tick=16
sz_title=20

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.plot(gamma,gap,'-o',label=r"gap (numerical)")
ax.plot(gamma,gamma,'-o',label=r"y=x")
#ax.plot(gamma,r*gamma,'-o',label='crude est.')
#ax.plot(gamma,r2*gamma,'-o',label='better est.')
#ax.plot(gamma,r22*gamma,'-o',label='2xbetter est.')

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlabel(r"$\Gamma$", fontsize=sz_label)
ax.set_ylabel(r"gap", fontsize=sz_label)

ax.tick_params(axis='x',direction='in', labelsize=sz_tick)
ax.tick_params(axis='y',direction='in', labelsize=sz_tick)
ax.tick_params(which='minor',axis='x', direction='in', labelsize=sz_tick)
ax.tick_params(which='minor',axis='y', direction='in', labelsize=sz_tick)

ax.legend(fontsize=sz_legend)

fig.set_size_inches(5.5,5)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#plt.savefig('/home/zuxin/Dropbox/Anderson_Holstein/article2/tex/figs/gap.png', dpi=600)

print(gap)
#print(gamma)

plt.show()


