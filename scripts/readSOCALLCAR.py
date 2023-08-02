import numpy as np

nproj = -1
ntyp = -1
nions = -1
sumionlmmax = [0]
ionlmmax = []
SPINORB_VKS = []
with open('SocAllCar', 'rb') as SocAllCar:
    tmplen = np.fromfile(SocAllCar, dtype = np.int32, count = 1)[0]
    ntyp, nions, ijmax, nproj = np.fromfile(SocAllCar, dtype = np.int32, count = 4)
    tmplen = np.fromfile(SocAllCar, dtype = np.int32, count = 1)[0]
    print('ntyp, nions, ijmax, nproj =', ntyp, nions, ijmax, nproj)
    for ityp in range(ntyp):
        tmplen = np.fromfile(SocAllCar, dtype = np.int32, count = 1)[0]
        num_ityp, lmmax_ityp = np.fromfile(SocAllCar, dtype = np.int32, count = 2)
        tmplen = np.fromfile(SocAllCar, dtype = np.int32, count = 1)[0]
        print('num_ityp, lmmax_ityp =', num_ityp, lmmax_ityp)
    for ia in range(nions):
        if ia > 0:
            sumionlmmax += [sumionlmmax[-1] + ionlmmax[-1]]
        tmplen = np.fromfile(SocAllCar, dtype = np.int32, count = 1)[0]
        ionlmmax += [np.fromfile(SocAllCar, dtype = np.int32, count = 1)[0]]
        tmplen = np.fromfile(SocAllCar, dtype = np.int32, count = 1)[0]
    print('ionlmmax, sumionlmmax =', ionlmmax, sumionlmmax)
    for iatom in range(nions):
        vks = []
        for iss in range(4):
            for ilm in range(ijmax):
                tmplen = np.fromfile(SocAllCar, dtype = np.int32, count = 1)[0]
                vks += [ np.fromfile(SocAllCar, dtype = complex, count = ijmax) ]
                tmplen = np.fromfile(SocAllCar, dtype = np.int32, count = 1)[0]
        SPINORB_VKS += [ np.array(vks, dtype = complex).reshape(4, ijmax, ijmax)[:, :ionlmmax[iatom], :ionlmmax[iatom]] ]
        print(SPINORB_VKS[-1].shape)


hsoc_full = []
with open('socallcar', 'rb') as socallcar:
    for iatom in range(nions):
        hsoc_full += [ np.fromfile(socallcar, dtype = complex,
                                   count = 4 * ionlmmax[iatom] * ionlmmax[iatom]
                                  ).reshape(4, ionlmmax[iatom], ionlmmax[iatom]) ]
        print(hsoc_full[-1].shape)

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['axes.linewidth'] = 1
fig, ax = plt.subplots(nrows = 2, ncols = 3,
                       sharex=False, sharey=False,
                       figsize = (6.0 * 3, 6.0 * 2))

for ia in range(nions):
    irow = ia // 3
    jcol = ia %  3
    #ax[irow][jcol].set_aspect('equal', adjustable='box', anchor='C')
    ax[irow][jcol].set_aspect('equal', adjustable='datalim')
    ax[irow][jcol].axis('square')
    #ax[irow][jcol].axis('image')
    
    for iss in range(0, 1):
        xdata = SPINORB_VKS[ia][iss].ravel()
        ydata = hsoc_full[ia][iss].ravel()
        print("xdata, ydata:", xdata.shape, ydata.shape)
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

fig.savefig('readSOCALLCAR.png', dpi=450, bbox_inches='tight')
