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
Tlims = [0.2, 200]

# plotting conventions
rs = 1.2    # plot image extent of +/- (rs * rout) for each target
plt.style.use('classic')
plt.rc('font', size=7)
left, right, bottom, top = 0.06, 0.94, 0.06, 0.98
wspace, hspace = 0.20, 0.25
fig = plt.figure(figsize=(7.5, 5.2))
gs = gridspec.GridSpec(3, 3)


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

    # mark the gap(s)
    rgap = disk.disk[targets[i]]['rgap']
    wgap = disk.disk[targets[i]]['wgap']
    for ir in range(len(rgap)):
        ax.fill_between([rgap[ir] - 2 * wgap[ir], rgap[ir] - 2 * wgap[ir],
                         rgap[ir] + 2 * wgap[ir], rgap[ir] + 2 * wgap[ir]],
                        [0.1, 1000, 1000, 0.1], color='darkgray', zorder=0)

    # mark the outer edge of the disk
    ax.plot([disk.disk[targets[i]]['rout'], disk.disk[targets[i]]['rout']], 
            [0.1, 1000], '--', color='darkgray', zorder=1)

    # limits and labeling
    Rlims = [0, 1.2]
    Tlims = [0.6, 200]
    ax.text(0.95, 0.93, disk.disk[targets[i]]['label'], 
            transform=ax.transAxes, ha='right', va='top')
    ax.set_xlim(Rlims)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    ax.set_ylim(Tlims)
    ax.set_yscale('log')
    ax.set_yticks([1, 10, 100])
    ax.set_yticklabels(['1', '10', '100'])
    if (i == 6):
        ax.set_xlabel('radius  ($^{\prime\prime}$)')
        ax.set_ylabel('$T \, _b$  (K)')
    #if not ((i == 0) or (i == 5)):
    #    ax.set_xticklabels([])
    #    ax.set_yticklabels([])
        

# adjust full figure layout
fig.subplots_adjust(wspace=wspace, hspace=hspace)
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
fig.savefig('../figs/frank_profiles.pdf')
