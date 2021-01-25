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
left, right, bottom, top = 0.06, 0.94, 0.07, 0.985
wspace, hspace = 0.20, 0.25
fig = plt.figure(figsize=(7.5, 4.5))
gs = gridspec.GridSpec(3, 3)

Tlims = [0.002, 20]
Rmax = [0.5, 0.5, 0.5, 0.6, 0.6, 0.7, 1.2, 1.2, 1.2]

# target loop (index i)
for i in range(len(targets)):

    # load data and residual images
    if np.logical_or(targets[i] == 'HD143006', targets[i] == 'HD163296'):
        dhdu = fits.open('data/'+targets[i]+'_data_symm.JvMcorr.fits')
        rhdu = fits.open('data/'+targets[i]+'_resid_symm.JvMcorr.fits')
    else:
        dhdu = fits.open('data/'+targets[i]+'_data.JvMcorr.fits')
        rhdu = fits.open('data/'+targets[i]+'_resid.JvMcorr.fits')
    dim, dhd = np.squeeze(dhdu[0].data), dhdu[0].header
    rim, rhd = np.squeeze(rhdu[0].data), rhdu[0].header
    dhdu.close()
    rhdu.close()

    # sky-frame Cartesian coordinates
    nx, ny = dhd['NAXIS1'], dhd['NAXIS2']
    RAo  = 3600 * dhd['CDELT1'] * (np.arange(nx) - (dhd['CRPIX1'] - 1))
    DECo = 3600 * dhd['CDELT2'] * (np.arange(ny) - (dhd['CRPIX2'] - 1))
    xs, ys = np.meshgrid(RAo - disk.disk[targets[i]]['dx'],
                         DECo - disk.disk[targets[i]]['dy'])

    # convert these to disk-frame polar coordinates
    inclr = np.radians(disk.disk[targets[i]]['incl'])
    PAr = np.radians(disk.disk[targets[i]]['PA'])
    xd = (xs * np.cos(PAr) - ys * np.sin(PAr)) / np.cos(inclr)
    yd = (xs * np.sin(PAr) + ys * np.cos(PAr))
    r, az = np.sqrt(xd**2 + yd**2), np.degrees(np.arctan2(yd, xd))


    # set location in figure
    ax = fig.add_subplot(gs[np.floor_divide(i, 3), i%3])

    # radial profile of residual / data pixel ratios
    dr = 0.005
    rbins = np.arange(dr, Rmax[i], dr)
    frac_ = np.abs(rim / dim)
    frac  = np.zeros_like(rbins)
    efrac = np.zeros_like(rbins)
    for ir in range(len(rbins)):
        inbin = ((r >= rbins[ir]-0.5*dr) & (r < rbins[ir]+0.5*dr))
        frac[ir] = np.average(frac_[inbin])
        efrac[ir] = np.std(frac_[inbin])

    # plot the residual / data ratio in each pixel, as a function of r
    ax.fill_between(rbins, frac-efrac, frac+efrac, color='r', alpha=0.2, 
                    edgecolor='none')
    ax.plot(rbins, frac, 'k')

    # mark the outer boundaries (2 x RMS)
    rout = disk.disk[targets[i]]['rout']
    ax.plot([rout, rout], Tlims, '--', color='slategray', zorder=0)

    # mark the Huang et al (2018) gaps
    for ig in range(len(D_huang[i])):
        rgh = D_huang[i] 
        ax.plot([rgh[ig] * 1e-3, rgh[ig] * 1e-3], Tlims, ':k', zorder=0)

    # mark the gap(s)
    rgap = disk.disk[targets[i]]['rgap']
    wgap = disk.disk[targets[i]]['wgap']
    for ir in range(len(rgap)):
        ax.fill_between([rgap[ir] - wgap[ir], rgap[ir] - wgap[ir],
                         rgap[ir] + wgap[ir], rgap[ir] + wgap[ir]],
                        [0.1, 1000, 1000, 0.1], color='silver', zorder=0)

    # limits and labeling
    ax.text(0.06, 0.91, disk.disk[targets[i]]['label'], 
            transform=ax.transAxes, ha='left', va='top')
    ax.set_xlim([0, Rmax[i]])
    ax.set_ylim(Tlims)
    ax.set_yscale('log')
    ax.set_yticks([0.01, 0.1, 1, 10])
    ax.set_yticklabels(['0.01', '0.1', '1', '10'])
    if (i == 6):
        ax.set_xlabel('radius  ($^{\prime\prime}$)')
        ax.set_ylabel('|fractional residual|')
        

# adjust full figure layout
fig.subplots_adjust(wspace=wspace, hspace=hspace)
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
fig.savefig('../figs/frac_resids.pdf')
