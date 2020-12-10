import os, sys, time
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from astropy.visualization import (LinearStretch, ImageNormalize)


# set color map
c2 = plt.cm.Reds(np.linspace(0, 1, 32))
c1 = plt.cm.Blues_r(np.linspace(0, 1, 32))
c1 = np.vstack([c1, np.ones((12, 4))])
colors = np.vstack((c1, c2))
mymap = mcolors.LinearSegmentedColormap.from_list('eddymap', colors)
cmap = mymap


# target-specific inputs
disk_name = ['dx5mas', 'dy5mas', 'incl_2lo', 'incl_2hi', 'PA_2lo', 'PA_2hi',
             'zr_0.01', 'zr_0.02', 'zr_0.03']
disk_lbls = ['dx 5mas', 'dy 5mas', 'incl -2', 'incl +2', 'PA -2', 'PA +2',
             'z/r 0.01', 'z/r 0.02', 'z/r 0.03']
offs = [0., 0.]
incl = 35.
PA = 110.
rbound = 150./140.
rout = 0.8
vspan = 200

# plotting conventions
rs = 1.5    # plot image extent of +/- (rs * rout) for each target
plt.style.use('default')
plt.rc('font', size=7)
left, right, bottom, top = 0.08, 0.93, 0.05, 0.99
wspace, hspace = 0.30, 0.15
fig = plt.figure(figsize=(7.5, 6.1))
gs = gridspec.GridSpec(3, 4, width_ratios=(1, 1, 1, 0.09),
                             height_ratios=(1, 1, 1))


# target loop (index i)
for i in range(len(disk_name)):

    ### Prepare image plotting
    # parse header information into physical numbers
    hd = fits.open('data/fiducial_'+disk_name[i]+'_resid.fits')[0].header
    nx, ny = hd['NAXIS1'], hd['NAXIS2']
    RAo  = 3600 * hd['CDELT1'] * (np.arange(nx) - (hd['CRPIX1'] - 1))
    DECo = 3600 * hd['CDELT2'] * (np.arange(ny) - (hd['CRPIX2'] - 1))
    dRA, dDEC = np.meshgrid(RAo - offs[0], DECo - offs[1])

    # beam parameters 
    bmaj, bmin, bPA = 3600 * hd['BMAJ'], 3600 * hd['BMIN'], hd['BPA']
    barea = (np.pi * bmaj * bmin / (4 * np.log(2))) / (3600 * 180 / np.pi)**2

    # image setups
    im_bounds = (dRA.max(), dRA.min(), dDEC.min(), dDEC.max())
    dRA_lims, dDEC_lims = [rs*rout, -rs*rout], [-rs*rout, rs*rout]

    # intensity limits, and stretch
    norm = ImageNormalize(vmin=-vspan, vmax=vspan, stretch=LinearStretch())

    # set location in figure
    ax = fig.add_subplot(gs[np.floor_divide(i, 3), i%3])

    # load image
    hdu = fits.open('data/fiducial_'+disk_name[i]+'_resid.fits')
    img = np.squeeze(hdu[0].data)

    # plot the image (in uJy / beam units)
    im = ax.imshow(1e6 * img, origin='lower', cmap=cmap, extent=im_bounds,
                   norm=norm, aspect='equal')

    # annotations
    tt = np.linspace(-np.pi, np.pi, 181)
    inclr, PAr = np.radians(incl), np.radians(PA)
    rgapi, rgapo = 70./140., 77./140.
    xgi, ygi = rgapi * np.cos(tt) * np.cos(inclr), rgapi * np.sin(tt)
    ax.plot(xgi * np.cos(PAr) + ygi * np.sin(PAr),
            -xgi * np.sin(PAr) + ygi * np.cos(PAr), '-', color='darkgray',
            lw=0.5, alpha=0.5)
    xgo, ygo = rgapo * np.cos(tt) * np.cos(inclr), rgapo * np.sin(tt)
    ax.plot(xgo * np.cos(PAr) + ygo * np.sin(PAr),
            -xgo * np.sin(PAr) + ygo * np.cos(PAr), '-', color='darkgray',
            lw=0.5, alpha=0.5)
    xb, yb = rbound * np.cos(tt) * np.cos(inclr), rbound * np.sin(tt)
    ax.plot(xb * np.cos(PAr) + yb * np.sin(PAr),
            -xb * np.sin(PAr) + yb * np.cos(PAr), ':', color='gray',
            lw=0.5, alpha=0.5)

    xgi, ygi = (15./140.) * np.cos(tt) * np.cos(inclr), (15./140.) * np.sin(tt)
    ax.plot(xgi * np.cos(PAr) + ygi * np.sin(PAr),
            -xgi * np.sin(PAr) + ygi * np.cos(PAr), '-', color='darkgray',
            lw=0.5, alpha=0.5)
    xgo, ygo = 18./140. * np.cos(tt) * np.cos(inclr), (18./140.) * np.sin(tt)
    ax.plot(xgo * np.cos(PAr) + ygo * np.sin(PAr),
            -xgo * np.sin(PAr) + ygo * np.cos(PAr), '-', color='darkgray',
            lw=0.5, alpha=0.5)



    # clean beams
    beam = Ellipse((dRA_lims[0] + 0.1*np.diff(dRA_lims),
                    dDEC_lims[0] + 0.1*np.diff(dDEC_lims)), bmaj, bmin, 90-bPA)
    beam.set_facecolor('k')
    ax.add_artist(beam)

    # limits and labeling
    ax.text(dRA_lims[0] + 0.05*np.diff(dRA_lims),
            dDEC_lims[1] - 0.13*np.diff(dDEC_lims), disk_lbls[i], color='k')
    ax.set_xlim(dRA_lims)
    ax.set_ylim(dDEC_lims)
    ax.set_xticks([1, 0, -1])
    ax.set_yticks([-1, 0, 1])
    if (i == 6):
        ax.set_xlabel('RA offset  (arcsec)')
        ax.set_ylabel('DEC offset  (arcsec)')
#    if np.logical_and((i !=0 ), (i != 5)):
#        ax.set_xticklabels([])
#        ax.set_yticklabels([])


# colorbar
cbax = fig.add_subplot(gs[:1,3])
cb = Colorbar(ax=cbax, mappable=im, orientation='vertical',
              ticklocation='right')	#, ticks=[0, 5, 10, 20, 30, 40])
#cbax.set_yticklabels(['0', '2', '5', '10', '20', '50'])
#cbax.tick_params('both', length=3, direction='in', which='major')
cb.set_label('residual  ($\mu$Jy / beam)', rotation=270, labelpad=10)


# adjust full figure layout
fig.subplots_adjust(wspace=wspace, hspace=hspace)
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
fig.savefig('test_resid_maps.pdf')
