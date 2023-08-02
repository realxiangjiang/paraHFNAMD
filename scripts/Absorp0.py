#!/usr/bin/env python
import numpy as np
import matplotlib as mpl
mpl.use('agg')
mpl.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors

########################################################################

def gradient_fill(x, y, fill_color=None, ax=None, direction=1, **kwargs):
  line, = ax.plot(x, y, **kwargs)
  if fill_color is None:
    fill_color = line.get_color()

  zorder = line.get_zorder()
  alpha = line.get_alpha()
  alpha = 1.0 if alpha is None else alpha

  z = np.empty((100, 1, 4), dtype=float)
  rgb = mcolors.colorConverter.to_rgb(fill_color)
  z[:, :, :3] = rgb
  if direction == 1:
      z[:, :, -1] = np.linspace(0, alpha, 100)[:, None]
  else:
      z[:, :, -1] = np.linspace(alpha, 0, 100)[:, None]

  xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
  im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                  origin='lower', zorder=zorder)

  xy = np.column_stack([x, y])
  if direction == 1:
      xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
  else:
      xy = np.vstack([[xmin, ymax], xy, [xmax, ymax], [xmin, ymax]])
  clip_path = Polygon(xy, lw=0.0, facecolor='none',
                      edgecolor='none', closed=True)
  ax.add_patch(clip_path)
  im.set_clip_path(clip_path)

  ax.autoscale(True)

  return line, im
########################################################################


def lorentz_smearing(x, x0, sigma=0.03):
  '''
  Lorentz smearing of a Delta function.
  '''
  return sigma / np.pi / ((x-x0)**2 + sigma**2)

def gaussian_smearing(x, x0, sigma=0.05):
  '''
  Gaussian smearing of a Delta function.
  '''
  smear = np.zeros(x.size)
  condition = np.abs(x - x0) < (sigma * 5.)
  smear[condition] = 1. / (np.sqrt(2*np.pi) * sigma) * \
      np.exp(-(x[condition] - x0)**2 / (2*sigma**2))

  return smear

def gaussian_smearing_org(x, x0, sigma=0.05):
  '''
  Gaussian smearing of a Delta function.
  '''

  return 1. / (np.sqrt(2*np.pi) * sigma) * np.exp(-(x - x0)**2 / (2*sigma**2))

def str2cplx(s):
  tmp = s.split(',')
  r = float(tmp[0].strip('('))
  i = float(tmp[1].strip(')'))
  return r + 1.0j*i

bseout = open('bseout', 'r')
lines = bseout.readlines()
noXct = int((len(lines)+1)/53)
enXct = np.zeros((noXct))
tdm = np.zeros((noXct,3), dtype=complex)
for ii in range(noXct):
  enXct[ii] = float(lines[ii*53].split()[-7])
  tdm[ii,:] = np.array(list(map(str2cplx,lines[ii*53].split()[-3:])), dtype=complex)
tdm = np.abs(tdm)**2
with open('eigenvalues.dat', 'w') as eig:
  for ii in range(noXct):
    eig.write(str(enXct[ii])+'  '+str(tdm[ii,:])+'\n')
emin = enXct.min()
emax = enXct.max()
eran = emax - emin
extra = 0.05
emin -= eran * extra
emax += eran * extra

nedos = 5000
xen = np.linspace(emin, emax, nedos)

sigma = 0.02
Jdos_smear = np.zeros((noXct, 3, nedos))
for ixct in range(noXct):
  x0 = enXct[ixct]
  for ixyz in range(3):
    Jdos_smear[ixct, ixyz] = gaussian_smearing_org(xen, x0, sigma)*tdm[ixct, ixyz]

Jdos = np.sum(Jdos_smear, axis=0)

########################################################################
fig = plt.figure()
fig.set_size_inches(4, 3)
ax = plt.subplot(111)

fill_direction = 1
lc = 'b'
which_xyz = 0 # polarization direction : 0 - x #
              #                          1 - y #
              #                          2 - z #
labels_xyz = ['x', 'y', 'z']
line, im = gradient_fill(xen, Jdos[which_xyz,:], ax = ax,
                        color=lc,
                        lw = 0.5,
                        direction=fill_direction)
ax.set_xlabel('Energy [eV]', fontsize='small',
              labelpad=5)
ax.set_ylabel('Abs [arb. unit]', fontsize='small',
              labelpad=10)
ax.tick_params(which='both', labelsize='small')

ax.set_xlim(emin, 2.5)
ax.set_ylim(0,100)
ax.set_yticks([])
fac = 10.0
ax.vlines(enXct, ymin=0.0, ymax=tdm[:,which_xyz] * fac, lw=1.0, color='k')

ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

plt.tight_layout(pad=0.50)
plt.savefig('Abs_'+labels_xyz[which_xyz]+'.png', dpi=360)
