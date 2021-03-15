import os, sys, time
import numpy as np
import scipy.constants as sc
from astropy.io import fits
sys.path.append('../')
import diskdictionary as disk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
from astropy.visualization import (AsinhStretch, LogStretch, LinearStretch, ImageNormalize)
import cmasher as cmr

# setups for data
grain_type = 'wscat'

kabs = 2.4
alb = 0.0

targets = ['SR4', 'RULup', 'Elias20', 'Sz129', 'Sz129', 'HD143006', 'HD143006',
           'GWLup', 'Elias24', 'HD163296', 'HD163296', 'AS209', 
           'AS209']
gap_ixs = [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1]

tlbls = ['SR 4', 'RU Lup', 'Elias 20', 'Sz 129', 'Sz 129', 'HD 143006',
         'HD 143006', 'GW Lup', 'Elias 24', 'HD 163296', 'HD 163296', 
         'AS 209', 'AS 209']
glbls = ['D11', 'D29', 'D25', 'D41', 'D64', 'D22', 'D51', 'D74', 'D57',
         'D48', 'D86', 'D61', 'D97']

zMp = np.array([2.16, 0.07, 0.05, 0.03, 999, 19.91, 0.33, 0.03, 0.84, 2.18, 0.14, 999, 0.65])

zlo = np.array([0.16, 0.25, 0.25, 0.25, 0, 0.16, 0.16, 0.17, 0.14, 0.14, 0.14, 0, 0.17])
zhi = np.array([0.13, 0.25, 0.25, 0.25, 0, 0.14, 0.14, 0.14, 0.16, 0.16, 0.16, 0, 0.14])

jMp = np.array([999, 999, 999, 999, 18.3, 999, 999, 3.6, 10., 999, 999, 999, 3.1])

plt.rcParams.update({'font.size': 8})

cmap = 'viridis_r'


# setup plot and grid
fig = plt.figure(figsize=(7.5, 7.5))
gs = gridspec.GridSpec(5, 3)

# loop
for i in range(len(targets)):

    # load the data
    dfile = targets[i]+'_gap'+str(gap_ixs[i])+'_'+grain_type+'.Mlims'
    dat = np.load('Mlims_data/'+dfile+'.npz')
    Mpl, Rth = dat['Mpl'], dat['Rth']
    Mcpd = np.log10(dat['Mlims'] * 0.01 * 5.974e27 / 1.898e30) 

    # set plot boundaries
    im_bounds = (np.log10(Mpl.min()), np.log10(Mpl.max()), 
                 Rth.min(), Rth.max())
    Mbnds = [np.log10(Mpl.min()), np.log10(Mpl.max())] 
    Rbnds = [0.1, 0.8]

    # panel assignment
    if (i < 11):
        ii = i
    else:
        ii = i + 1
    ax = fig.add_subplot(gs[np.floor_divide(ii, 3), ii%3])

    # plot the CPD masses
    im = ax.imshow(Mcpd.T, cmap=cmap, origin='lower',
                   extent=im_bounds, vmin=-5.0, vmax=-3.2, aspect='auto')

    # axes and annotations
    ax.set_xlim(Mbnds)
    ax.set_ylim(Rbnds)
    ax.text(0.03, 0.13, tlbls[i] + '\n' + glbls[i], ha='left', va='center', 
            transform=ax.transAxes, fontsize=7)
    if (ii == 12):
        ax.set_xlabel('$\log{(M_p \,\, / \,\, M_{\\rm Jup})}$')
        ax.set_ylabel('$R_{\\rm cpd} \,\, / \,\, R_{\\rm H}$')

    # plot the Rcpd = resolution curve (higher = not point-like anymore)
    apl = disk.disk[targets[i]]['rgap'][gap_ixs[i]] * \
          disk.disk[targets[i]]['distance']
    mstar = disk.disk[targets[i]]['mstar']
    Rhill = apl * (Mpl * 1.898e30 / (3. * mstar * 1.989e33))**(1./3.)
    if np.logical_or((targets[i] == 'HD143006'), (targets[i] == 'HD163296')):
        dfil = '../CSD_modeling/data/deep_'+targets[i]+'_data_symm.JvMcorr.fits'
    else:
        dfil = '../CSD_modeling/data/deep_'+targets[i]+'_data.JvMcorr.fits'
    hd = fits.open(dfil)[0].header
    res = 3600 * np.sqrt(hd['BMAJ'] * hd['BMIN']) * \
          disk.disk[targets[i]]['distance']
    ax.plot(np.log10(Mpl), 0.5 * res/Rhill, ':k', lw=0.9)

    # plot the Rcpd = gap width curve (higher = CPD bigger than gap)
    wgap = 0.5 * disk.disk[targets[i]]['wgap'][gap_ixs[i]] * \
           disk.disk[targets[i]]['distance'] * 2.355
    ax.plot(np.log10(Mpl), wgap / Rhill, 'm', lw=0.9)

    # find locations where tau > 1
    #rfrac = np.tile(Rth, (len(Mpl), 1)).T
    #Mtau = (8 * np.pi * (1-alb) * rfrac**2 * (Rhill*1.496e13)**2 / (5 * kabs)) 
    #logMtau = np.log10(Mtau / 5.974e27)
    #thick = np.ones_like(Mtau)
    #thick[logMtau >= Mcpd.T] = 0
    #ax.contour(np.log10(Mpl), Rth, thick, [1.0], colors='k')

    # mark the Zhang et al. (2018) regions
    mlo, mhi = np.log10(zMp[i]) - zlo[i], np.log10(zMp[i]) + zhi[i]
    ax.plot([mlo, mhi, mhi, mlo, mlo], [0.5, 0.5, 0.05, 0.05, 0.5], '--r', 
            lw=0.8)

    # mark the Joquera et al. (2021) regions
    ax.fill_between([np.log10(jMp[i]), Mbnds[1]], [Rbnds[1], Rbnds[1]], 
                    hatch='////', edgecolor='ghostwhite', facecolor='none',
                    linewidth=0)


    # find Mcpd / Mp values
    if (zMp[i] < 50):
        olapM = np.where((np.log10(Mpl) >= np.log10(zMp[i]) - zlo[i]) & \
                         (np.log10(Mpl) <= np.log10(zMp[i]) + zhi[i]))[0]
        olapR = np.where((Rth >= 0.1) & (Rth <= 0.5))[0]
        Mcpd_slice = Mcpd[olapM[0]:(olapM[-1]+1), olapR[0]:(olapR[-1]+1)]
        Mp_slice = np.tile(Mpl[olapM[0]:(olapM[-1]+1)], (len(olapR), 1)).T
        Mrat = 100 * (100 * 10**Mcpd_slice / Mp_slice)
        if not np.all(np.isnan(Mrat)):
            print('%10a  gap%i  %.2f  %.2f' % \
                  (targets[i], gap_ixs[i], np.nanmin(Mrat), np.nanmax(Mrat)))

# annotations / key
aax = fig.add_axes([0.69, 0.12, 0.94-0.69, 0.28])

#aax.fill_between([0.02, 0.1, 0.1, 0.02, 0.02], [0.98, 0.98, 0.8, 0.8, 0.98], 
#                 hatch='////', edgecolor='k', facecolor='none', lw=0)
aax.fill_between([0.02, 0.1], [0.98, 0.98], [0.8, 0.8], hatch='////', 
                 edgecolor='k', facecolor='none', lw=0)
aax.text(0.15, 0.89, 'Jorquera et al. 2021 \nexcluded planets', ha='left',
         va='center')

aax.plot([0.02, 0.1, 0.1, 0.02, 0.02], [0.73, 0.73, 0.55, 0.55, 0.73], '--r')
aax.text(0.15, 0.64, 'Zhang et al. 2018 \nplanet properties', ha='left', 
         va='center')

aax.plot([0.02, 0.10], [0.45, 0.45], ':k')
aax.text(0.15, 0.45, 'DSHARP angular resolution', ha='left', va='center')

aax.plot([0.02, 0.1], [0.3, 0.3], 'm')
aax.text(0.15, 0.30, 'inferred gap width', ha='left', va='center')

aax.set_xlim([0, 1])
aax.set_ylim([0, 1])
aax.axis('off')


# colorbar
cbax = fig.add_axes([0.69, 0.12, 0.94-0.69, 0.025])
cb = Colorbar(ax=cbax, mappable=im, orientation='horizontal', 
              ticklocation='bottom')
cb.set_label('$\log{(M_{\\rm cpd} \,\, / \,\, M_{\\rm Jup})}$')#, fontsize=12)

fig.subplots_adjust(wspace=0.25, hspace=0.25)
fig.subplots_adjust(left=0.06, right=0.94, bottom=0.05, top=0.99)
fig.savefig('../figs/Mcpd_limits_'+grain_type+'.pdf')
