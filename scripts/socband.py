#!/usr/bin/env python

import os
import numpy as np

def GetLattice(inFile = 'OUTCAR'):
    """
    extract reciprocal lattice from OUTCAR
    """

    outcar = [line for line in open(inFile) if line.strip()]

    for ii, line in enumerate(outcar):
        if 'NKPTS =' in line:
            nkpts = int(line.split()[3])
            nbnds = int(line.split()[-1])
        
        elif 'NELECT =' in line:
            nelect = int(float(line.split()[2]))

        elif 'reciprocal lattice vectors' in line:
            ibasis = ii + 1

        elif 'E-fermi' in line:
            efermi = float(line.split()[2])
            break

    # basis vector of reciprocal lattice
    B = np.array([line.split()[3:] for line in outcar[ibasis:ibasis+3]], 
                 dtype=float)

    return efermi, nelect, nkpts, nbnds, B

def GetBands(inFile = 'socout'):
    """
    extract band energy from socout
    """
    energy = []
    vkpts = []

    socout = [line for line in open(inFile) if line.strip()]
    for line in socout:
        if 'kpoint' in line:
            vkpts += [line.split()[4:]]
        elif 'band No.' in line:
            continue
        else:
            energy += [line.split()[1:]]

    vkpts = np.array(vkpts, dtype=float)
    energy = np.array(energy, dtype=float).reshape((nkpts, nbnds * 2, -1))

    # get band path
    vkpt_diff = np.diff(vkpts, axis=0)

    kpFile = open('KPOINTS')
    kpFile.readline()
    nkpts_per_line = int(kpFile.readline().split()[0])
    kpFile.close()

    flag = np.arange(nkpts_per_line - 1, np.size(vkpt_diff, axis=0), nkpts_per_line)
    #print(flag)
    vkpt_diff[flag] = 0.0
    kpt_path = np.zeros(nkpts, dtype=float)
    kpt_path[1:] = np.cumsum(np.linalg.norm(np.dot(vkpt_diff, B), axis=1))
    kpt_path /= kpt_path[-1]

    # get boundaries of band path
    xx = np.append(np.diff(kpt_path), 0.5)
    kpt_bounds = kpt_path[np.isclose(xx, 0.0)]

    # print kpt_bounds
    # print kpt_path.size, energy.shape

    return kpt_bounds, kpt_path, energy

################################################################################

efermi, nelect, nkpts, nbnds, B = GetLattice()
kpt_bounds, kpt_path, energy = GetBands()

# set energy zeros to Fermi energy
#energy -= efermi
energy[:, :, :2] -= np.max(energy[:, nelect - 1, 1])
        
################################################################################
# The Plotting part
################################################################################
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

mpl.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(nrows=1, ncols=1,
                       sharex=False, sharey=False,
                       figsize = (4, 6))
fig.subplots_adjust(left=0.12, right=0.95,
                    bottom=0.08, top=0.95,
                    wspace=0.10, hspace=0.10)

from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

spinxyz = 'z'

if spinxyz == 'x' or spinxyz == 'y' or spinxyz == 'z':
    if spinxyz == 'x':
        iz = 3
    elif spinxyz == 'y':
        iz = 4
    else:
        iz = 5
    EnergyWeight = energy[:, :, iz]
    norm = mpl.colors.Normalize(vmin=EnergyWeight.min(),
                                vmax=EnergyWeight.max())
    s_m = mpl.cm.ScalarMappable(cmap='bwr', norm=norm)
    s_m.set_array([EnergyWeight])
    
    for jj in np.arange(nbnds * 2):
        ax.plot(kpt_path, energy[:, jj, 0], '--',
                color='k', lw=1.0, alpha=0.6)

        #ax.plot(kpt_path, energy[:, jj, 1], '-',
        #        color='blue', lw=1.0, alpha=0.9)
        x = kpt_path
        y = energy[:, jj, 1]
        z = EnergyWeight[:, jj]

        DELTA = 0.3
        LW = 2.0
        ax.plot(x, y,
                lw=LW + 2 * DELTA,
                color='gray', zorder=1)

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments,
                            # cmap=opts.occLC_cmap, # alpha=0.7,
                            colors=[s_m.to_rgba(ww)
                                    for ww in (z[1:] + z[:-1])/2.]
                            # norm=plt.Normalize(0, 1)
                            )
        # lc.set_array((z[1:] + z[:-1]) / 2)
        lc.set_linewidth(LW)
        ax.add_collection(lc)

    divider = make_axes_locatable(ax)
    ax_cbar = divider.append_axes('right', size='3%', pad=0.02)
    cbar = plt.colorbar(s_m, cax=ax_cbar, orientation='vertical')
else:
    for jj in np.arange(nbnds * 2):
        ax.plot(kpt_path, energy[:, jj, 0], '--',
                color='red', lw=1.0, alpha=0.6)

        ax.plot(kpt_path, energy[:, jj, 1], '-',
                color='blue', lw=1.0, alpha=0.9)

for bd in kpt_bounds:
    ax.axvline(x=bd, ls='--', color='k', lw=0.5, alpha=0.6)

ax.axhline(y=0.0, ls=':', color='g', lw=0.5, alpha=0.6)

ax.set_ylim(-3, 3)
ax.set_xlim(0, 1)
ax.tick_params(which='both', labelsize='small')
ax.set_ylabel('Energy [eV]', fontsize='medium')

pos = [0,] + list(kpt_bounds) + [1,]
ax.set_xticks(pos)

#kpts_name =[xx for xx in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'][:len(pos)]
kpts_name = [r'$\Gamma$', 'K', 'M', r'$\Gamma$', 'A', 'L', 'H', 'A|L', 'M|K', 'H']
ax.set_xticklabels(kpts_name[:len(pos)], x=pos, fontsize='small')                   

fig.savefig('socband.png', dpi=360, bbox_inches='tight')
