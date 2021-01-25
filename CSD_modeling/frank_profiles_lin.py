import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import fits
sys.path.append('../')
import diskdictionary as disk

# target-specific inputs
targets = ['SR4', 'RULup', 'Elias20', 'Sz129', 'HD143006',
           'GWLup', 'Elias24', 'HD163296', 'AS209']

D_huang = [ [82],
            [90, 132, 183, 260],
            [181.6, 239],
            [255, 400],
            [133, 309],
            [479.2, 665],
            [418, 654],
            [100, 480, 855, 1450],
            [71.8, 197, 289.6, 503, 742.6, 872, 1132] ]

# plotting conventions
rs = 1.2    # plot image extent of +/- (rs * rout) for each target
plt.style.use('classic')
plt.rc('font', size=7)
left, right, bottom, top = 0.05, 0.95, 0.07, 0.985
wspace, hspace = 0.20, 0.25
fig = plt.figure(figsize=(7.5, 4.5))
gs = gridspec.GridSpec(3, 3)

Tmax = [50, 50, 50, 20, 20, 50, 50, 50, 50]
Rmax = [0.5, 0.5, 0.5, 0.6, 0.6, 0.7, 1.2, 1.2, 1.2]

# target loop (index i)
for i in range(len(targets)):

    # load frankenstein radial brightness profile fit
    if np.logical_or(targets[i] == 'HD143006', targets[i] == 'HD163296'):
        pfile = 'fits/' + targets[i]+'_symm_frank_profile_fit.txt'
    else:
        pfile = 'fits/' + targets[i] + '_frank_profile_fit.txt'
    r_frank, Inu_frank, eInu_frank = np.loadtxt(pfile).T

    # get frequency
    hd = fits.open('data/'+targets[i]+'_resid.JvMcorr.fits')[0].header
    freq = hd['CRVAL3']

    # convert to brightness temperatures (R-J limit)
    kB_, c_ = 1.38064852e-16, 2.99792e10
    Tb_frank = c_**2 * 1e-23 * Inu_frank / (2 * kB_ * freq**2)
    eTb_frank = c_**2 * 1e-23 * eInu_frank / (2 * kB_ * freq**2)

    # set location in figure
    ax = fig.add_subplot(gs[np.floor_divide(i, 3), i%3])

    # plot the radial profile
    ax.fill_between(r_frank, Tb_frank-eTb_frank, Tb_frank+eTb_frank, 
                    color='r')

    # mark the Huang et al (2018) gaps
    for ig in range(len(D_huang[i])):
        rgh = D_huang[i] 
        ax.plot([rgh[ig] * 1e-3, rgh[ig] * 1e-3], [0, Tmax[i]], ':k', zorder=0)

    # mark the gap(s)
    rgap = disk.disk[targets[i]]['rgap']
    wgap = disk.disk[targets[i]]['wgap']
    for ir in range(len(rgap)):
        ax.fill_between([rgap[ir] - wgap[ir], rgap[ir] - wgap[ir],
                         rgap[ir] + wgap[ir], rgap[ir] + wgap[ir]],
                        [0.1, 1000, 1000, 0.1], color='silver', zorder=0)

    # limits and labeling
    ax.text(0.94, 0.91, disk.disk[targets[i]]['label'], 
            transform=ax.transAxes, ha='right', va='top')
    ax.set_xlim([0, Rmax[i]])
    ax.set_ylim([0, Tmax[i]])
    if (i == 6):
        ax.set_xlabel('radius  ($^{\prime\prime}$)')
        ax.set_ylabel('$T \, _b$  (K)')
        

# adjust full figure layout
fig.subplots_adjust(wspace=wspace, hspace=hspace)
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
fig.savefig('../figs/frank_profiles_lin.pdf')
