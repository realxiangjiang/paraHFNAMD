import numpy as np
nkpts = 100
nbnds = 24
twonbnds = nbnds * 2
socmat = []
for ncore in [1, 4]:
    with open('socmat_%i' % ncore) as inf:
        socmat += [np.fromfile(inf, dtype = complex, 
                               count = nkpts * twonbnds * twonbnds)]
socmat = np.array(socmat).reshape(2, nkpts, twonbnds, twonbnds)

inputkpts = range(0,6)
#is1 = 0; is2 = 0
#socmat = socmat[:, inputkpts, is1 * nbnds:(is1 + 1) * nbnds,
#                              is2 * nbnds:(is2 + 1) * nbnds]
socmat = socmat[:, inputkpts]
print(socmat.shape)

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['axes.linewidth'] = 1
fig, ax = plt.subplots(nrows = 2, ncols = 3,
                       sharex=False, sharey=False,
                       figsize = (6.0 * 3, 6.0 * 2))
ikpt = 0
ibnd = 0
for ik in range(6):
    irow = ik // 3
    jcol = ik %  3
    #ax[irow][jcol].set_aspect('equal', adjustable='box', anchor='C')
    ax[irow][jcol].set_aspect('equal', adjustable='datalim')
    ax[irow][jcol].axis('square')
    #ax[irow][jcol].axis('image')
    
    xdata = socmat[0, ik].ravel()
    ydata = socmat[1, ik].ravel()
    print("ik =", inputkpts[ik] + 1, xdata.shape, ydata.shape)
    ax[irow][jcol].axline([0.00, 0.00], [0.01, 0.01], color='grey')
    ax[irow][jcol].plot(np.real(xdata), np.real(ydata), lw = 0,
                        marker = 'o', ms = 4.0, color = 'red')
    ax[irow][jcol].plot(np.imag(xdata), np.imag(ydata), lw = 0,
                        marker = 'o', ms = 4.0, color = 'blue')
    #ax[irow][jcol].plot(np.abs(xdata), np.abs(ydata), lw = 0,
    #                    marker = 'o', ms = 4.0, color = 'green')

    #lmdata = [(l, m) for l in allL[ia] for m in range(-l, l+1)]
    #for lm, x, y in zip(lmdata, np.abs(xdata.ravel()), np.abs(ydata.ravel())):
    #    ax[irow][jcol].text(x, y, '({},{})'.format(*lm), ha="center", va='center', fontsize='small')

fig.savefig('readSOCMAT.png', dpi=450, bbox_inches='tight')
