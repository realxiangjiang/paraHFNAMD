import numpy as np
totnspns = 1
totnkpts = 100
totnbnds = 24
projphi = []
numatoms = -1
kpoints = -1
bands = -1
lmmax = -1
with open('NORMCAR', 'rb') as normcar:
    numatoms = np.fromfile(normcar, dtype = np.int32, count = 1)[0]
    lmmax = np.fromfile(normcar, dtype = np.int32, count = numatoms)
    nspns = np.fromfile(normcar, dtype = np.int32, count = 1)[0]
    spins = np.fromfile(normcar, dtype = np.int32, count = nspns)
    nkpts = np.fromfile(normcar, dtype = np.int32, count = 1)[0]
    kpoints = np.fromfile(normcar, dtype = np.int32, count = nkpts)
    nbnds = np.fromfile(normcar, dtype = np.int32, count = 1)[0]
    bands = np.fromfile(normcar, dtype = np.int32, count = nbnds)
    print('numatoms =', numatoms)
    print('lmmax =', lmmax)
    print('spins =', nspns, spins)
    print('kpoints =', nkpts, kpoints)
    print('bands =', nbnds, bands)
    for iatom in range(numatoms):
        projphi += [np.fromfile(normcar, dtype = complex, 
                                count = nspns * (nkpts * nbnds) * lmmax[iatom]).reshape(nspns, nkpts, nbnds, lmmax[iatom])]
        #print(projphi[-1][:20])
print('kpoints:', kpoints)
print('bands:', bands)
for ia in range(numatoms):
    print(projphi[ia].shape)

nproj = -1
nions = -1
sumionlmmax = [0]
ionlmmax = []
with open('SocAllCar', 'rb') as socallcar:
    tmplen = np.fromfile(socallcar, dtype = np.int32, count = 1)[0]
    ntyp, nions, ijmax, nproj = np.fromfile(socallcar, dtype = np.int32, count = 4)
    tmplen = np.fromfile(socallcar, dtype = np.int32, count = 1)[0]
    print('ntyp, nions, ijmax, nproj =', ntyp, nions, ijmax, nproj)
    for ityp in range(ntyp):
        tmplen = np.fromfile(socallcar, dtype = np.int32, count = 1)[0]
        num_ityp, lmmax_ityp = np.fromfile(socallcar, dtype = np.int32, count = 2)
        tmplen = np.fromfile(socallcar, dtype = np.int32, count = 1)[0]
        print('num_ityp, lmmax_ityp =', num_ityp, lmmax_ityp)
    for ia in range(nions):
        if ia > 0:
            sumionlmmax += [sumionlmmax[-1] + ionlmmax[-1]]
        tmplen = np.fromfile(socallcar, dtype = np.int32, count = 1)[0]
        ionlmmax += [np.fromfile(socallcar, dtype = np.int32, count = 1)[0]]
        tmplen = np.fromfile(socallcar, dtype = np.int32, count = 1)[0]
    print('ionlmmax, sumionlmmax =', ionlmmax, sumionlmmax)

CPROJ = []
with open('NormalCAR', 'rb') as vaspnmc:
    sizeofdouble = 8
    sizeofint    = 4
    sizeofcd     = 16
    tmplen = np.fromfile(vaspnmc, dtype = np.int32, count = 1)[0]
    lmdim, nions_loc, nrspinors = np.fromfile(vaspnmc, dtype = np.int32, count = 3)
    tmplen = np.fromfile(vaspnmc, dtype = np.int32, count = 1)[0]
    print('lmdim, nions_loc, nrspinors =', lmdim, nions_loc, nrspinors)

    tmplen = np.fromfile(vaspnmc, dtype = np.int32, count = 1)[0]
    vaspnmc.seek(tmplen + sizeofint, 1) #skip CQIJ
    print(tmplen, lmdim * lmdim * nions_loc * nrspinors * sizeofdouble)
    tmplen = np.fromfile(vaspnmc, dtype = np.int32, count = 1)[0]
    nprod, npro, ntyp = np.fromfile(vaspnmc, dtype = np.int32, count = 3)
    tmplen = np.fromfile(vaspnmc, dtype = np.int32, count = 1)[0]
    print('nprod, npro, ntyp =', npro, npro, ntyp)
    for i in range(ntyp):
        tmplen = np.fromfile(vaspnmc, dtype = np.int32, count = 1)[0]
        ilmmax, nityp = np.fromfile(vaspnmc, dtype = np.int32, count = 2)
        tmplen = np.fromfile(vaspnmc, dtype = np.int32, count = 1)[0]
        print(i + 1, tmplen // sizeofint, ': ilmmax, nityp =', lmmax, nityp)
    for ispn in range(totnspns):
        for ikpt in range(totnkpts):
            for ibnd in range(totnbnds):
                tmplen = np.fromfile(vaspnmc, dtype = np.int32, count = 1)[0]
                CPROJ += [ np.fromfile(vaspnmc, dtype = complex, count = nproj) ]
                tmplen = np.fromfile(vaspnmc, dtype = np.int32, count = 1)[0]

cproj = np.array(CPROJ, dtype = complex).reshape(totnspns, totnkpts, totnbnds, nproj)
#print(cproj[:, 0, 0, 20:30])
#CPROJ = np.array(CPROJ, dtype = complex).transpose().reshape(nproj * totnspns, totnkpts * totnbnds)
#CPROJ = CPROJ.transpose().reshape(totnkpts * totnbnds, nproj, totnspns).transpose()
#CPROJ = CPROJ.reshape(totnspns, nproj, totnkpts, totnbnds)
#print(cproj.shape, CPROJ.shape)
'''
for ispn in range(totnspns):
    for ikpt in range(totnkpts):
        for ibnd in range(totnbnds):
            for iproj in range(nproj):
                print(cproj[ispn, ikpt, ibnd, iproj] - CPROJ[ispn, iproj, ikpt, ibnd])
                '''
ion_cproj = []
for ia in range(nions):
    ion_cproj += [ cproj[:, :, :, sumionlmmax[ia]:sumionlmmax[ia] + ionlmmax[ia]] ]
for ia in range(nions):
    print(ion_cproj[ia].shape)

allL = [[0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 0, 0, 1, 1],
        [2, 2, 0, 0, 1, 1]]

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['axes.linewidth'] = 1
fig, ax = plt.subplots(nrows = 2, ncols = 3,
                       sharex=False, sharey=False,
                       figsize = (6.0 * 3, 6.0 * 2))
ikpt = 5
ibnd = 0
for ia in range(nions):
    irow = ia // 3
    jcol = ia %  3
    #ax[irow][jcol].set_aspect('equal', adjustable='box', anchor='C')
    ax[irow][jcol].set_aspect('equal', adjustable='datalim')
    ax[irow][jcol].axis('square')
    #ax[irow][jcol].axis('image')
    
    xdata = ion_cproj[ia][:, kpoints][:, :, bands]
    ydata = projphi[ia]
    xdata = xdata[:, ikpt]#, ibnd]
    ydata = ydata[:, ikpt]#, ibnd]
    print("ia =", ia, xdata.shape, ydata.shape)
    ax[irow][jcol].axline([0.00, 0.00], [0.01, 0.01], color='grey')
    ax[irow][jcol].plot(np.real(xdata.ravel()), np.real(ydata.ravel()), lw = 0,
                        marker = 'o', ms = 4.0, color = 'red')
    ax[irow][jcol].plot(np.imag(xdata.ravel()), np.imag(ydata.ravel()), lw = 0,
                        marker = 'o', ms = 4.0, color = 'blue')
    #ax[irow][jcol].plot(np.abs(xdata.ravel()), np.abs(ydata.ravel()), lw = 0,
    #                    marker = 'o', ms = 4.0, color = 'green')

    #lmdata = [(l, m) for l in allL[ia] for m in range(-l, l+1)]
    #for lm, x, y in zip(lmdata, np.abs(xdata.ravel()), np.abs(ydata.ravel())):
    #    ax[irow][jcol].text(x, y, '({},{})'.format(*lm), ha="center", va='center', fontsize='small')

fig.savefig('readNORMCAR.png', dpi=450, bbox_inches='tight')
