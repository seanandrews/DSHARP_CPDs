import os, sys, time
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from astropy.visualization import (LinearStretch, ImageNormalize)
from deproject_vis import deproject_vis
sys.path.append('../')
import diskdictionary as disk


# set color map
c2 = plt.cm.Reds(np.linspace(0, 1, 32))
c1 = plt.cm.Blues_r(np.linspace(0, 1, 32))
c1 = np.vstack([c1, np.ones((26, 4))])
colors = np.vstack((c1, c2))
mymap = mcolors.LinearSegmentedColormap.from_list('eddymap', colors)
cmap = mymap


# target-specific inputs
targets = ['SR4', 'RULup', 'Elias20', 'Sz129', 'HD143006',
           'GWLup', 'Elias24', 'HD163296', 'AS209']

xyticks = [[-0.2, 0.0, 0.2],
           [-0.4, -0.2, 0.0, 0.2, 0.4],
           [-0.5, 0.0, 0.5],
           [-0.5, 0.0, 0.5],
           [-0.5, 0.0, 0.5],
           [-0.5, 0.0, 0.5],
           [-1, 0.0, 1],
           [-1, 0.0, 1.0],
           [-1.0, 0.0, 1.0]]
 
vspan = 10


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

    # load residual image / header information
    if np.logical_or((targets[i] == 'HD143006'), (targets[i] == 'HD163296')):
        hdu = fits.open('data/deep_'+targets[i]+'_resid_symm.JvMcorr.fits')
    else:
        hdu = fits.open('data/deep_'+targets[i]+'_resid.JvMcorr.fits')
    img = 1e6 * np.squeeze(hdu[0].data) / disk.disk[targets[i]]['RMS']
    hd = hdu[0].header
    hdu.close()

    # sky-plane Cartesian coordinates
    nx, ny = hd['NAXIS1'], hd['NAXIS2']
    RAo  = 3600 * hd['CDELT1'] * (np.arange(nx) - (hd['CRPIX1'] - 1))
    DECo = 3600 * hd['CDELT2'] * (np.arange(ny) - (hd['CRPIX2'] - 1))
    xs, ys = np.meshgrid(RAo - disk.disk[targets[i]]['dx'], 
                         DECo - disk.disk[targets[i]]['dy'])

    # disk-plane Cartesian coordinates
    inclr = np.radians(disk.disk[targets[i]]['incl'])
    PAr = np.radians(disk.disk[targets[i]]['PA'])
    xd = (xs * np.cos(PAr) - ys * np.sin(PAr)) / np.cos(inclr)
    yd = (xs * np.sin(PAr) + ys * np.cos(PAr))

    # disk-plane Polar coordinates
    rd  = np.sqrt(xd**2 + yd**2)
    azd = np.degrees(np.arctan2(yd, xd))

    # beam parameters 
    bmaj, bmin, bPA = 3600 * hd['BMAJ'], 3600 * hd['BMIN'], hd['BPA']
    barea = (np.pi * bmaj * bmin / (4 * np.log(2))) / (3600 * 180 / np.pi)**2

    # image setups
    rout = disk.disk[targets[i]]['rout']
    im_bounds = (xs.max(), xs.min(), ys.min(), ys.max())
    xs_lims, ys_lims = [rs*rout, -rs*rout], [-rs*rout, rs*rout]

    # set location in figure
    ax = fig.add_subplot(gs[np.floor_divide(i, 3), i%3])

    # plot the image (in uJy / beam units)
    im = ax.imshow(img, origin='lower', cmap=cmap, extent=im_bounds,
                   vmin=-vspan, vmax=vspan, aspect='equal')

    # define, mark gaps; derive scatter in gaps
    tt = np.linspace(-np.pi, np.pi, 91)
    rgap = disk.disk[targets[i]]['rgap']	#[::-1]
    wgap = disk.disk[targets[i]]['wgap']	#[::-1]
    wbm = np.sqrt(bmaj * bmin) / 2.355
    gcols = ['dimgrey', 'k']	#['k', 'dimgrey']
    for ir in range(len(rgap)):
        # inner gap boundary
        xgi = (rgap[ir] - wgap[ir] - wbm) * np.cos(tt) * np.cos(inclr) 
        ygi = (rgap[ir] - wgap[ir] - wbm) * np.sin(tt)
        ax.plot( xgi * np.cos(PAr) + ygi * np.sin(PAr),
                -xgi * np.sin(PAr) + ygi * np.cos(PAr), '-', color=gcols[ir],
                lw=0.8, alpha=0.8)

        # outer gap boundary
        xgo = (rgap[ir] + wgap[ir] + wbm) * np.cos(tt) * np.cos(inclr)
        ygo = (rgap[ir] + wgap[ir] + wbm) * np.sin(tt)
        ax.plot( xgo * np.cos(PAr) + ygo * np.sin(PAr),
                -xgo * np.sin(PAr) + ygo * np.cos(PAr), '-', color=gcols[ir],
                lw=0.8, alpha=0.8)

        # Apply a boolean mask to isolate the search annulus
        mask = np.zeros_like(img, dtype='bool')
        bndi, bndo = (rd >= (rgap[ir]-wgap[ir])), (rd <= (rgap[ir]+wgap[ir]))
        mask[np.logical_and(bndi, bndo)] = 1
        RMS_gap = np.std(img[mask] * disk.disk[targets[i]]['RMS'])

        # Get typical RMS_gap from recoveries
        rfil = '../CPD_search/recoveries/' + \
               targets[i]+'_gap'+str(ir)+'_recoveries.all.txt'
        if os.path.exists(rfil):
            Fi, Fr, mdl, ri, rr, azi, azr, xr, yr, mu, rms = np.loadtxt(rfil).T
            med_RMS = np.mean(rms)
        else:
            med_RMS = 0
        print('%10s  gap%i  %.1f  %.1f  %.1f  %.3f' % \
              (targets[i], ir, RMS_gap, med_RMS, disk.disk[targets[i]]['RMS'],
               med_RMS / disk.disk[targets[i]]['RMS']))
        

    # mark Rout
    xb, yb = rout * np.cos(tt) * np.cos(inclr), rout * np.sin(tt)
    ax.plot( xb * np.cos(PAr) + yb * np.sin(PAr),
            -xb * np.sin(PAr) + yb * np.cos(PAr), '--', color='gray',
            lw=0.5, alpha=0.5)

    # clean beams
    beam = Ellipse((xs_lims[0] + 0.1*np.diff(xs_lims),
                    ys_lims[0] + 0.1*np.diff(ys_lims)), bmaj, bmin, 90-bPA)
    beam.set_facecolor('k')
    ax.add_artist(beam)

    # limits and labeling
    ax.text(xs_lims[0] + 0.05*np.diff(xs_lims),
            ys_lims[1] - 0.09*np.diff(ys_lims), 
            disk.disk[targets[i]]['label'], color='k')
    ax.text(xs_lims[1] - 0.03*np.diff(xs_lims),
            ys_lims[0] + 0.05*np.diff(ys_lims),
            str(np.int(np.round(disk.disk[targets[i]]['RMS']))), 
            color='dimgrey', ha='right', va='center')
    ax.set_xlim(xs_lims)
    ax.set_ylim(ys_lims)
    ax.set_xticks((xyticks[i])[::-1])
    ax.set_yticks(xyticks[i])
    if (i == 6):
        ax.set_xlabel('RA offset  ($^{\prime\prime}$)')
        ax.set_ylabel('DEC offset  ($^{\prime\prime}$)', labelpad=5)


# colorbar
cbax = fig.add_axes([right + 0.01, 0.5*(top+bottom)-0.25, 0.02, 0.50])
cb = Colorbar(ax=cbax, mappable=im, orientation='vertical',
              ticklocation='right', 
              ticks=[-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
cb.set_label('residual S/N', rotation=270, labelpad=8)


# adjust full figure layout
fig.subplots_adjust(wspace=wspace, hspace=hspace)
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
fig.savefig('../figs/resid_maps.pdf')
