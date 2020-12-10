import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import fits

# target-specific inputs
disk_name = ['SR4', 'RULup', 'Elias20', 'Sz129', 'HD143006',
             'GWLup', 'Elias24', 'HD163296', 'AS209']
disk_lbls = ['SR 4', 'RU Lup', 'Elias 20', 'Sz 129', 'HD 143006',
             'GW Lup', 'Elias 24', 'HD 163296', 'AS 209']
rgapi = [0.060, 0.165, 0.17, 0.22, 0.08,
         0.44, 0.34, 0.39, 0.07]
rgapo = [0.095, 0.205, 0.195, 0.28, 0.21,
         0.52, 0.52, 0.62, 0.09]
rbound = [0.25, 0.42, 0.48, 0.48, 0.52,
          0.63, 1.05, 1.23, 0.25]
routs = 0.8
routl = 0.8
Tlims = [0.2, 200]

# plotting conventions
rs = 1.5    # plot image extent of +/- (rs * rout) for each target
plt.style.use('classic')
plt.rc('font', size=7)
left, right, bottom, top = 0.06, 0.94, 0.06, 0.98
wspace, hspace = 0.20, 0.25
fig = plt.figure(figsize=(7.5, 5.2))
gs = gridspec.GridSpec(3, 3)


# target loop (index i)
for i in range(len(disk_name)):

    # load frankenstein radial brightness profile fit
    pfile = 'fits/' + disk_name[i] + '_frank_profile_fit.txt'
    r_frank, Inu_frank, eInu_frank = np.loadtxt(pfile).T

    # get frequency
    hd = fits.open('data/'+disk_name[i]+'_resid.fits')[0].header
    freq = hd['CRVAL3']

    # convert to brightness temperatures (R-J limit)
    kB_, c_ = 1.38064852e-16, 2.99792e10
    Tb_frank = c_**2 * 1e-23 * Inu_frank / (2 * kB_ * freq**2)
    eTb_frank = c_**2 * 1e-23 * eInu_frank / (2 * kB_ * freq**2)

    # set location in figure
    ax = fig.add_subplot(gs[np.floor_divide(i, 3), i%3])

    # plot the radial profile
    ax.fill_between(r_frank, Tb_frank-eTb_frank, Tb_frank+eTb_frank, 
                    color='dodgerblue')

    mdl = 15 * ((r_frank * 140) / 10)**-0.5
    mdl[r_frank*140 >= 90.] = (15 * (90./10.)**-0.5) * ((r_frank[r_frank*140 >= 90] * 140) / 90)**-5
    ax.plot(r_frank, mdl, 'r')

    # mark the gap(s)
    ax.fill_between([rgapi[i], rgapi[i], rgapo[i], rgapo[i]],
                    [0.1, 1000, 1000, 0.1], color='darkgray', zorder=0)
    if (i == 4):
        ax.fill_between([0.27, 0.27, 0.36, 0.36], [0.1, 1000, 1000, 0.1],
                        color='darkgray', zorder=0)
    if (i == 7):
        ax.fill_between([0.73, 0.73, 0.95, 0.95], [0.1, 1000, 1000, 0.1],
                        color='darkgray', zorder=0)

    # mark the outer edge of the disk
    ax.plot([rbound[i], rbound[i]], [0.1, 1000], '--', color='darkgray',
            zorder=1)

    # limits and labeling
    if (i < 5):
        rout = routs
        Tlo = 0.6
    else: 
        rout = routl
        Tlo = 0.6
    Rlims = [0, rs * rout]
    Tlims = [Tlo, 200]
    ax.text(0.95, 0.93, disk_lbls[i], transform=ax.transAxes, ha='right', 
            va='top')
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
fig.savefig('figs/frank_profiles.pdf')
