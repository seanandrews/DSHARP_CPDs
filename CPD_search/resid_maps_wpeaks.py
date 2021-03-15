import os, sys, time
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse, Circle
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
 
vspan = 5


# plotting conventions
rs = 0.8    # plot image extent of +/- (rs * rout) for each target
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
    ddir = '../CSD_modeling/data/deep_'
    if np.logical_or((targets[i] == 'HD143006'), (targets[i] == 'HD163296')):
        hdu = fits.open(ddir+targets[i]+'_resid_symm.JvMcorr.fits')
    else:
        hdu = fits.open(ddir+targets[i]+'_resid.JvMcorr.fits')
    hd = hdu[0].header
    img = 1e6 * np.squeeze(hdu[0].data) / disk.disk[targets[i]]['RMS']
    hdu.close()

    # sky-plane Cartesian coordinates
    nx, ny = hd['NAXIS1'], hd['NAXIS2']
    RAo  = 3600 * hd['CDELT1'] * (np.arange(nx) - (hd['CRPIX1'] - 1))
    DECo = 3600 * hd['CDELT2'] * (np.arange(ny) - (hd['CRPIX2'] - 1))
    xs, ys = np.meshgrid(RAo - disk.disk[targets[i]]['dx'], 
                         DECo - disk.disk[targets[i]]['dy'])

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

    # annotations
    tt = np.linspace(-np.pi, np.pi, 91)
    inclr = np.radians(disk.disk[targets[i]]['incl'])
    PAr = np.radians(disk.disk[targets[i]]['PA'])
    rgap = disk.disk[targets[i]]['rgap']#[::-1]
    wgap = disk.disk[targets[i]]['wgap']#[::-1]
    gcols = ['k', 'dimgrey']
    for ir in [0]:	#range(len(rgap)):
        # mark gap boundaries
        xgi = (rgap[ir] - wgap[ir]) * np.cos(tt) * np.cos(inclr) 
        ygi = (rgap[ir] - wgap[ir]) * np.sin(tt)
        ax.plot( xgi * np.cos(PAr) + ygi * np.sin(PAr),
                -xgi * np.sin(PAr) + ygi * np.cos(PAr), '-', color=gcols[ir],
                lw=0.8, alpha=0.8)
        xgo = (rgap[ir] + wgap[ir]) * np.cos(tt) * np.cos(inclr)
        ygo = (rgap[ir] + wgap[ir]) * np.sin(tt)
        ax.plot( xgo * np.cos(PAr) + ygo * np.sin(PAr),
                -xgo * np.sin(PAr) + ygo * np.cos(PAr), '-', color=gcols[ir],
                lw=0.8, alpha=0.8)

        # mark *peak* location
        pkr = disk.disk[targets[i]]['peakr']
        pkaz = disk.disk[targets[i]]['peakaz']
        xpk = 1e-3 * pkr[ir] * np.sin(np.radians(pkaz[ir])) * np.sin(PAr) + \
              1e-3 * pkr[ir] * np.cos(np.radians(pkaz[ir])) * np.cos(inclr) * \
              np.cos(PAr)
        ypk = 1e-3 * pkr[ir] * np.sin(np.radians(pkaz[ir])) * np.cos(PAr) - \
              1e-3 * pkr[ir] * np.cos(np.radians(pkaz[ir])) * np.cos(inclr) * \
              np.sin(PAr)
        loc_peak = Circle((xpk, ypk), 
                          0.02*np.diff(ys_lims), color='limegreen', lw=0.5)
        loc_peak.set_facecolor('none')
        ax.add_artist(loc_peak)
        rpk  = np.sqrt(xpk**2 + ypk**2)
        PApk = 90. - np.degrees(np.arctan2(ypk, xpk))
        print(targets[i])
        print(xpk, ypk, rpk, PApk)
        print(' ')
  
    # mark Rout
    xb, yb = rout * np.cos(tt) * np.cos(inclr), rout * np.sin(tt)
    ax.plot( xb * np.cos(PAr) + yb * np.sin(PAr),
            -xb * np.sin(PAr) + yb * np.cos(PAr), '--', color='gray',
            lw=0.5, alpha=0.5)

    ds = 0.2
    xs_lims = [xpk+ds, xpk-ds]
    ys_lims = [ypk-ds, ypk+ds]

    # clean beams
    beam = Ellipse((xs_lims[0] + 0.1*np.diff(xs_lims),
                    ys_lims[0] + 0.1*np.diff(ys_lims)), bmaj, bmin, 90-bPA)
    beam.set_facecolor('k')
    ax.add_artist(beam)

    # limits and labeling
    ax.text(xs_lims[0] + 0.05*np.diff(xs_lims),
            ys_lims[1] - 0.09*np.diff(ys_lims), 
            disk.disk[targets[i]]['label'], color='k')
    ax.set_xlim([xpk+ds, xpk-ds])
    ax.set_ylim([ypk-ds, ypk+ds])
    #ax.set_xticks((xyticks[i])[::-1])
    #ax.set_yticks(xyticks[i])
    if (i == 6):
        ax.set_xlabel('RA offset  ($^{\prime\prime}$)')
        ax.set_ylabel('DEC offset  ($^{\prime\prime}$)', labelpad=8)


# colorbar
cbax = fig.add_axes([right + 0.01, 0.5*(top+bottom)-0.25, 0.02, 0.50])
cb = Colorbar(ax=cbax, mappable=im, orientation='vertical',
              ticklocation='right', 
              ticks=[-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])	#, ticks=[0, 5, 10, 20, 30, 40])
#cbax.set_yticklabels(['0', '2', '5', '10', '20', '50'])
#cbax.tick_params('both', length=3, direction='in', which='major')
cb.set_label('residual S/N', rotation=270, labelpad=10)


# adjust full figure layout
fig.subplots_adjust(wspace=wspace, hspace=hspace)
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
fig.savefig('../figs/resid_maps_wpeaks.pdf')
