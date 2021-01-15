import os, sys, time
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
from matplotlib.patches import Ellipse
from astropy.visualization import (AsinhStretch, ImageNormalize)
sys.path.append('../')
import diskdictionary as disk


# set color map
cmap = 'inferno'


# target-specific inputs
targets = ['SR4', 'RULup', 'Elias20', 'Sz129', 'HD143006',
           'GWLup', 'Elias24', 'HD163296', 'AS209']

xyticks = [[-0.2, 0.0, 0.2],
           [-0.4, -0.2, 0.0, 0.2, 0.4],
           [-0.5, 0.0, 0.5],
           [-0.5, 0.0, 0.5],
           [-0.5, 0.0, 0.5],
           [-0.5, 0.0, 0.5],
           [-1.0, -0.5, 0.0, 0.5, 1.0],
           [-1, 0.0, 1.0],
           [-1.0, 0.0, 1.0]]
 
# constants
c_ = 2.99792e10
k_ = 1.38057e-16


# plotting conventions
rs = 1.2    # plot image extent of +/- (rs * rout) for each target
plt.style.use('default')
plt.rc('font', size=7)
left, right, bottom, top = 0.06, 0.91, 0.05, 0.99
wspace, hspace = 0.25, 0.25
fig = plt.figure(figsize=(7.5, 6.5))
gs = gridspec.GridSpec(3, 3, width_ratios=(1, 1, 1), height_ratios=(1, 1, 1))


# target loop (index i)
for i in range(len(targets)):

    ### Prepare image plotting
    # parse header information into physical numbers
    if np.logical_or((targets[i] == 'HD143006'), (targets[i] == 'HD163296')):
        hd = fits.open('data/'+targets[i]+'_data_symm.JvMcorr.fits')[0].header
    else:
        hd = fits.open('data/'+targets[i]+'_data.JvMcorr.fits')[0].header
    nx, ny = hd['NAXIS1'], hd['NAXIS2']
    RAo  = 3600 * hd['CDELT1'] * (np.arange(nx) - (hd['CRPIX1'] - 1))
    DECo = 3600 * hd['CDELT2'] * (np.arange(ny) - (hd['CRPIX2'] - 1))
    dRA, dDEC = np.meshgrid(RAo - disk.disk[targets[i]]['dx'], 
                            DECo - disk.disk[targets[i]]['dy'])
    freq = hd['CRVAL3']

    # beam parameters 
    bmaj, bmin, bPA = 3600 * hd['BMAJ'], 3600 * hd['BMIN'], hd['BPA']
    barea = (np.pi * bmaj * bmin / (4 * np.log(2))) / (3600 * 180 / np.pi)**2

    # image setups
    rout = disk.disk[targets[i]]['rout']
    im_bounds = (dRA.max(), dRA.min(), dDEC.min(), dDEC.max())
    dRA_lims, dDEC_lims = [rs*rout, -rs*rout], [-rs*rout, rs*rout]

    # intensity limits, and stretch
    norm = ImageNormalize(vmin=0, vmax=disk.disk[targets[i]]['maxTb'], 
                          stretch=AsinhStretch())

    # set location in figure
    ax = fig.add_subplot(gs[np.floor_divide(i, 3), i%3])

    # load image
    if np.logical_or((targets[i] == 'HD143006'), (targets[i] == 'HD163296')):
        hdu = fits.open('data/'+targets[i]+'_data_symm.JvMcorr.fits')
    else:
        hdu = fits.open('data/'+targets[i]+'_data.JvMcorr.fits')
    img = np.squeeze(hdu[0].data) 
    Tb = (1e-23 * img / barea) * c_**2 / (2 * k_ * freq**2)

    # plot the image (in K units)
    im = ax.imshow(Tb, origin='lower', cmap=cmap, extent=im_bounds,
                   norm=norm, aspect='equal')

    # annotations
    tt = np.linspace(-np.pi, np.pi, 181)
    inclr = np.radians(disk.disk[targets[i]]['incl'])
    PAr = np.radians(disk.disk[targets[i]]['PA'])
    gcols = ['g', 'g']
    rgap = disk.disk[targets[i]]['rgap']
    wgap = disk.disk[targets[i]]['wgap']
    wbm = np.sqrt(bmaj * bmin) / 2.355
    for ir in range(len(rgap)):
        xgi = (rgap[ir] - wgap[ir] - wbm) * np.cos(tt) * np.cos(inclr)
        ygi = (rgap[ir] - wgap[ir] - wbm) * np.sin(tt)
        ax.plot( xgi * np.cos(PAr) + ygi * np.sin(PAr),
                -xgi * np.sin(PAr) + ygi * np.cos(PAr), '-', color=gcols[ir],
                lw=0.5)
        xgo = (rgap[ir] + wgap[ir] + wbm) * np.cos(tt) * np.cos(inclr)
        ygo = (rgap[ir] + wgap[ir] + wbm) * np.sin(tt)
        ax.plot( xgo * np.cos(PAr) + ygo * np.sin(PAr),
                -xgo * np.sin(PAr) + ygo * np.cos(PAr), '-', color=gcols[ir],
                lw=0.5)

    # clean beams
    beam = Ellipse((dRA_lims[0] + 0.1*np.diff(dRA_lims),
                    dDEC_lims[0] + 0.1*np.diff(dDEC_lims)), bmaj, bmin, 90-bPA)
    beam.set_facecolor('w')
    ax.add_artist(beam)

    # limits and labeling
    ax.text(dRA_lims[0] + 0.05*np.diff(dRA_lims),
            dDEC_lims[1] - 0.09*np.diff(dDEC_lims), 
            disk.disk[targets[i]]['label'], color='w')
    ax.set_xlim(dRA_lims)
    ax.set_ylim(dDEC_lims)
    ax.set_xticks((xyticks[i])[::-1])
    ax.set_yticks(xyticks[i])
    if (i == 6):
        ax.set_xlabel('RA offset  ($^{\prime\prime}$)')
        ax.set_ylabel('DEC offset  ($^{\prime\prime}$)')


# adjust full figure layout
fig.subplots_adjust(wspace=wspace, hspace=hspace)
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
fig.savefig('../figs/data_maps.pdf')
