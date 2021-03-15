import os, sys, time           
import numpy as np             
from astropy.io import fits      
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
from matplotlib.patches import Ellipse
from astropy.visualization import (AsinhStretch, ImageNormalize)
from deproject_vis import deproject_vis
sys.path.append('../')
import diskdictionary as disk

# set color map
cmap = 'inferno'

# constants
c_ = 2.99792e10
k_ = 1.38057e-16

# target-specific inputs
targets = ['GWLup', 'Elias24', 'HD163296', 'AS209']
Tbticks = [[0, 2, 5, 10, 15, 20],
           [0, 5, 10, 20, 30, 40, 50],
           [0, 5, 10, 20, 30, 50, 70],
           [0, 5, 10, 20, 30]]

# plotting conventions
rout = 0.80
rs = 1.5    # plot image extent of +/- (rs * rout) for each target
dmr = ['data', 'model', 'resid']
plt.style.use('default')
plt.rc('font', size=6)
left, right, bottom, top = 0.09, 0.92, 0.07, 0.985
wspace, hspace = 0.03, 0.15
fig = plt.figure(figsize=(3.5, 4.4))
gs = gridspec.GridSpec(4, 4, width_ratios=(1, 1, 1, 0.08),
                             height_ratios=(1, 1, 1, 1))

# target loop (index i)
for i in range(len(targets)):

    ### Prepare image plotting
    # parse header information into physical numbers
    if (targets[i] == 'HD163296'):
        dfile = 'data/deep_'+targets[i]+'_data_symm.JvMcorr.fits'
    else:
        dfile = 'data/deep_'+targets[i]+'_data.JvMcorr.fits'
    hd = fits.open(dfile)[0].header
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
    im_bounds = (dRA.max(), dRA.min(), dDEC.min(), dDEC.max())
    dRA_lims, dDEC_lims = [rs*rout, -rs*rout], [-rs*rout, rs*rout]

    # intensity limits, and stretch
    norm = ImageNormalize(vmin=0., vmax=disk.disk[targets[i]]['maxTb'], 
                          stretch=AsinhStretch())


    ### Loop through data, model, residual images
    for j in range(len(dmr)):
        # load image
        if (targets[i] == 'HD163296'):
            dfile = 'data/deep_'+targets[i]+'_'+dmr[j]+'_symm.JvMcorr.fits'
        else:
            dfile = 'data/deep_'+targets[i]+'_'+dmr[j]+'.JvMcorr.fits'
        hdu = fits.open(dfile)
        img = np.squeeze(hdu[0].data)    

        # set location in figure
        ax = fig.add_subplot(gs[i,j])

        # plot the image (in brightness temperature units)
        Tb = (1e-23 * img / barea) * c_**2 / (2 * k_ * freq**2)
        im = ax.imshow(Tb, origin='lower', cmap=cmap, extent=im_bounds,
                       norm=norm, aspect='equal')

        # clean beams
        beam = Ellipse((dRA_lims[0] + 0.1*np.diff(dRA_lims),
                        dDEC_lims[0] + 0.1*np.diff(dDEC_lims)), 
                       bmaj, bmin, 90-bPA)
        beam.set_facecolor('w')
        ax.add_artist(beam)

        # limits and labeling
        if (j == 0):
            ax.text(dRA_lims[0] + 0.03*np.diff(dRA_lims), 
                    dDEC_lims[1] - 0.10*np.diff(dDEC_lims), 
                    disk.disk[targets[i]]['label'], color='w', fontsize=6)
        ax.set_xlim(dRA_lims)
        ax.set_ylim(dDEC_lims)
        ax.set_yticks([-1, 0.0, 1])
        if (i == 3) and (j == 0):
            ax.set_xlabel('RA offset  ($^{\prime\prime}$)', labelpad=2.5)
            ax.set_ylabel('DEC offset  ($^{\prime\prime}$)', labelpad=2)
            ax.tick_params(axis='y', length=1.5)
            ax.tick_params(axis='x', length=2)
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.axis('off')


    ### Colormap scalebar
    cbax = fig.add_subplot(gs[i,3])
    cbax.tick_params(axis='y', length=2)
    cb = Colorbar(ax=cbax, mappable=im, orientation='vertical',
                  ticklocation='right', ticks=Tbticks[i])
    if (i == 3):
        cb.set_label('$T_b$  (K)', rotation=270, labelpad=7)


# adjust full figure layout
fig.subplots_adjust(wspace=wspace, hspace=hspace)
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
fig.savefig('../figs/dmrs2.pdf')
