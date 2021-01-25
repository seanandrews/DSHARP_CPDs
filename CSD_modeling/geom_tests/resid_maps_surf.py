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
c1 = np.vstack([c1, np.ones((26, 4))])
colors = np.vstack((c1, c2))
mymap = mcolors.LinearSegmentedColormap.from_list('eddymap', colors)
cmap = mymap


# target-specific inputs
disk_name = ['zr_0.03', 'zr_0.05', 'zr3_dx5', 'zr3_dy5', 'zr3_incl2', 'zr3_PA5']
disk_lbls = ['$z/r$=0.03', '$z/r$=0.05', '$\\Delta x$=+5, $z/r$=0.03',
             '$\\Delta y$=+5, $z/r$=0.03', '$\\Delta i$=+2, $z/r$=0.03',
             '$\\Delta {\\rm PA}$=+5, $z/r$=0.03']
offs = [0., 0.]
incl = 35.
PA = 110.
rbound = 150./140.
rout = 0.8
vspan = 10

# plotting conventions
rs = 1.5    # plot image extent of +/- (rs * rout) for each target
plt.style.use('default')
plt.rc('font', size=7)
left, right, bottom, top = 0.10, 0.85, 0.079, 0.985
wspace, hspace = 0.03, 0.03
fig = plt.figure(figsize=(3.5, 4.37))
gs = gridspec.GridSpec(3, 2, width_ratios=(1, 1),
                             height_ratios=(1, 1, 1))


# target loop (index i)
for i in range(len(disk_name)):
    print(disk_name[i])

    ### Prepare image plotting
    # parse header information into physical numbers
    hd = fits.open('data/'+disk_name[i]+'_resid.JvMcorr.fits')[0].header
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
    ax = fig.add_subplot(gs[np.floor_divide(i, 2), i%2])

    # load image
    hdu = fits.open('data/'+disk_name[i]+'_resid.JvMcorr.fits')
    img = 1e6 * np.squeeze(hdu[0].data) / 6.

    # plot the image (in uJy / beam units)
    im = ax.imshow(img, origin='lower', cmap=cmap, extent=im_bounds,
                   norm=norm, aspect='equal')

    # annotations
    tt = np.linspace(-np.pi, np.pi, 181)
    inclr, PAr = np.radians(incl), np.radians(PA)
    rgapi, rgapo = 70./140., 82./140.
    xgi, ygi = rgapi * np.cos(tt) * np.cos(inclr), rgapi * np.sin(tt)
    ax.plot(xgi * np.cos(PAr) + ygi * np.sin(PAr),
            -xgi * np.sin(PAr) + ygi * np.cos(PAr), '-', color='k',
            lw=0.5, alpha=0.5)
    xgo, ygo = rgapo * np.cos(tt) * np.cos(inclr), rgapo * np.sin(tt)
    ax.plot(xgo * np.cos(PAr) + ygo * np.sin(PAr),
            -xgo * np.sin(PAr) + ygo * np.cos(PAr), '-', color='k',
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
    ax.text(dRA_lims[0] + 0.04*np.diff(dRA_lims),
            dDEC_lims[1] - 0.10*np.diff(dDEC_lims), disk_lbls[i], color='k',
            fontsize=6)
    ax.set_xlim(dRA_lims)
    ax.set_ylim(dDEC_lims)
    if (i == 4):
        ax.set_xlabel('RA offset  ($^{\prime\prime}$)')
        ax.set_ylabel('DEC offset  ($^{\prime\prime}$)', labelpad=0.5)
        ax.set_xticks([1, 0, -1])
        ax.set_yticks([-1, 0, 1])
    else:
        ax.set_xticks([])
        ax.set_yticks([])


# colorbar
cbax = fig.add_axes([right + 0.01, bottom, 0.02, 0.5*(top + bottom)-bottom])
cb = Colorbar(ax=cbax, mappable=im, orientation='vertical',
              ticklocation='right', 
              ticks=[-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
cb.set_label('residual S/N', rotation=270, labelpad=8)


# adjust full figure layout
fig.subplots_adjust(wspace=wspace, hspace=hspace)
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
fig.savefig('../../figs/geom_test2.pdf')
