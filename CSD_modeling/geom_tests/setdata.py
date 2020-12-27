import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from astropy.io import fits
from astropy.visualization import (AsinhStretch, LinearStretch, ImageNormalize)
from frank.radial_fitters import FrankFitter
from frank.geometry import FixedGeometry
from frank.io import save_fit



# controls
target = 'dx5_incl2'
#target = 'dx5_PA5'
#target = 'incl2_PA5'
#target = 'incl2_PA5_dx5'
#target = 'zr3_dx5'
#target = 'zr3_dy5'
#target = 'zr3_incl2'
#target = 'zr3_PA5'

frank  = True
im_dat = False
im_res = True
im_mdl = False
annotate_res = False



# constants
c_ = 2.99792e10
k_ = 1.38057e-16

# residuals color map
c2 = plt.cm.Reds(np.linspace(0, 1, 32))
c1 = plt.cm.Blues_r(np.linspace(0, 1, 32))
c1 = np.vstack([c1, np.ones((12, 4))])
colors = np.vstack((c1, c2))
mymap = mcolors.LinearSegmentedColormap.from_list('eddymap', colors)


# crude passing mechanism
f = open('whichdisk.txt', 'w')
f.write(target)
f.close()


### - IMAGE THE DATA
if im_dat:
    print('....')
    print('Imaging the data')
    print('....')
    os.system('casa --nogui --nologger --nologfile -c data_imaging.py')
    print('....')
    print('Finished imaging the data')
    print('....')


### - PLOT THE ANNOTATED IMAGE

# load data
dhdu = fits.open('data/dx1mas_data.JvMcorr.fits')
dimg, hd = np.squeeze(dhdu[0].data), dhdu[0].header

# parse coordinate frame indices into physical numbers
RA = 3600 * hd['CDELT1'] * (np.arange(hd['NAXIS1']) - (hd['CRPIX1'] - 1)) 
DEC = 3600 * hd['CDELT2'] * (np.arange(hd['NAXIS2']) - (hd['CRPIX2'] - 1))
dRA, dDEC = np.meshgrid(RA, DEC)
freq = hd['CRVAL3']

# disk-frame polar coordinates
inclr = np.radians(35.)	
PAr = np.radians(110.)
xd = (dRA * np.cos(PAr) - dDEC * np.sin(PAr)) / np.cos(inclr)
yd = (dRA * np.sin(PAr) + dDEC * np.cos(PAr))
r, theta = np.sqrt(xd**2 + yd**2), np.degrees(np.arctan2(yd, xd))

# beam parameters
bmaj, bmin, bPA = 3600 * hd['BMAJ'], 3600 * hd['BMIN'], hd['BPA']
beam_area = (np.pi * bmaj * bmin / (4 * np.log(2))) / (3600 * 180 / np.pi)**2

# image setups
rout = 1.1
im_bounds = (dRA.max(), dRA.min(), dDEC.min(), dDEC.max())
dRA_lims, dDEC_lims = [1.5*rout, -1.5*rout], [-1.5*rout, 1.5*rout]

# intensity limits, and stretch
norm = ImageNormalize(vmin=0, vmax=50., stretch=AsinhStretch())
cmap = 'inferno'

### Plot the data image
plt.style.use('default')
fig = plt.figure(figsize=(7.0, 5.9))
gs  = gridspec.GridSpec(1, 2, width_ratios=(1, 0.04))

# image (sky-plane)
ax = fig.add_subplot(gs[0,0])
Tb = (1e-23 * dimg / beam_area) * c_**2 / (2 * k_ * freq**2)
im = ax.imshow(Tb, origin='lower', cmap=cmap, extent=im_bounds, 
               norm=norm, aspect='equal')

# annotations
tbins = np.linspace(-np.pi, np.pi, 181)

rgapi = [0.15, 0.70]
rgapo = [0.18, 0.82]
for ir in range(len(rgapi)):
    rgi = rgapi[ir]
    rgo = rgapo[ir]
    xgi, ygi = rgi * np.cos(tbins) * np.cos(inclr), rgi * np.sin(tbins)
    ax.plot( xgi * np.cos(PAr) + ygi * np.sin(PAr),
            -xgi * np.sin(PAr) + ygi * np.cos(PAr), ':w')
    xgo, ygo = rgo * np.cos(tbins) * np.cos(inclr), rgo * np.sin(tbins)
    ax.plot( xgo * np.cos(PAr) + ygo * np.sin(PAr),
            -xgo * np.sin(PAr) + ygo * np.cos(PAr), ':w')

xout, yout = rout * np.cos(tbins) * np.cos(inclr), rout * np.sin(tbins)
ax.plot( xout * np.cos(PAr) + yout * np.sin(PAr),
        -xout * np.sin(PAr) + yout * np.cos(PAr), '--w')

# beam
beam = Ellipse((dRA_lims[0] + 0.1*np.diff(dRA_lims), 
                dDEC_lims[0] + 0.1*np.diff(dDEC_lims)), bmaj, bmin, 90-bPA)
beam.set_facecolor('w')
ax.add_artist(beam)

# limits, labeling
ax.set_xlim(dRA_lims)
ax.set_ylim(dDEC_lims)
ax.set_xlabel('RA offset  ($^{\prime\prime}$)')
ax.set_ylabel('DEC offset  ($^{\prime\prime}$)')

# add a scalebar
cbax = fig.add_subplot(gs[:,1])
cb = Colorbar(ax=cbax, mappable=im, orientation='vertical',
              ticklocation='right')
cb.set_label('brightness temperature  (K)', rotation=270, labelpad=22)

# adjust layout
fig.subplots_adjust(wspace=0.02)
fig.subplots_adjust(left=0.11, right=0.89, bottom=0.1, top=0.98)
fig.savefig('../../figs/'+target+'_dataimage.pdf')




### - FRANK VISIBILITY MODELING 
if frank:
    print('....')
    print('Performing visibility modeling')
    print('....')
    # load the visibility data
    dat = np.load('data/'+target+'.vis.npz')
    u, v, vis, wgt = dat['u'], dat['v'], dat['Vis'], dat['Wgt']

    # set the disk viewing geometry
    geom = FixedGeometry(35., 110., 0.0, 0.0)

    # configure the fitting code setup
    FF = FrankFitter(Rmax=2*rout, geometry=geom, 
                     N=300, alpha=1.3, weights_smooth=0.1)

    # fit the visibilities
    sol = FF.fit(u, v, vis, wgt)

    # save the fit
    save_fit(u, v, vis, wgt, sol, prefix='fits/'+target)
    print('....')
    print('Finished visibility modeling')
    print('....')



### Imaging
if im_res:
    print('....')
    print('Imaging residuals')
    print('....')
    os.system('casa --nogui --nologger --nologfile -c resid_imaging.py')
    print('....')
    print('Finished imaging residuals')
    print('....')

if im_mdl:
    print('....')
    print('Imaging model')
    print('....')
    os.system('casa --nogui --nologerr --nologfile -c model_imaging.py') 
    print('....')
    print('Finished imaging model')
    print('....')


### +/- Residual plot
if os.path.exists('data/'+target+'_resid.JvMcorr.fits'):
    print('....')
    print('Making residual +/- plot')
    print('using file created on: %s' % \
          time.ctime(os.path.getctime('data/'+target+'_resid.JvMcorr.fits')))
    print('....')

    # load residual image
    rhdu = fits.open('data/'+target+'_resid.JvMcorr.fits')
    rimg = np.squeeze(rhdu[0].data)

    # set up plot
    plt.style.use('classic')
    fig = plt.figure(figsize=(7.0, 5.9))
    gs  = gridspec.GridSpec(1, 2, width_ratios=(1, 0.04))

    # image (sky-plane)
    ax = fig.add_subplot(gs[0,0])
    vmin, vmax = -50, 50    # these are in microJy/beam units
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())
    im = ax.imshow(1e6*rimg, origin='lower', cmap=mymap, extent=im_bounds, 
                   norm=norm, aspect='equal')

    # gap markers
    gcols = ['k', 'darkgray']
    for ir in range(len(rgapi)):
        rgi = rgapi[ir]
        rgo = rgapo[ir]
        xgi, ygi = rgi * np.cos(tbins) * np.cos(inclr), rgi * np.sin(tbins)
        ax.plot( xgi * np.cos(PAr) + ygi * np.sin(PAr),
                -xgi * np.sin(PAr) + ygi * np.cos(PAr), gcols[ir])
        xgo, ygo = rgo * np.cos(tbins) * np.cos(inclr), rgo * np.sin(tbins)
        ax.plot( xgo * np.cos(PAr) + ygo * np.sin(PAr),
                -xgo * np.sin(PAr) + ygo * np.cos(PAr), gcols[ir])

    # outer edge marker
    xout, yout = rout * np.cos(tbins) * np.cos(inclr), rout * np.sin(tbins)
    ax.plot( xout * np.cos(PAr) + yout * np.sin(PAr),
            -xout * np.sin(PAr) + yout * np.cos(PAr), '--', color='darkgray')

    # beam
    beam = Ellipse((dRA_lims[0] + 0.1*np.diff(dRA_lims), 
                    dDEC_lims[0] + 0.1*np.diff(dDEC_lims)), bmaj, bmin, 90-bPA)
    beam.set_facecolor('k')
    ax.add_artist(beam)

    # limits, labeling
    ax.set_xlim(dRA_lims)
    ax.set_ylim(dDEC_lims)
    ax.set_xlabel('RA offset  ($^{\prime\prime}$)')
    ax.set_ylabel('DEC offset  ($^{\prime\prime}$)')

    # add a scalebar
    cbax = fig.add_subplot(gs[:,1])
    cb = Colorbar(ax=cbax, mappable=im, orientation='vertical',
                  ticklocation='right') 
    cb.set_label('residual brightness  ($\mu$Jy / beam)', rotation=270, 
                 labelpad=18)

    # adjust layout
    fig.subplots_adjust(wspace=0.02)
    fig.subplots_adjust(left=0.11, right=0.89, bottom=0.1, top=0.98)
    fig.savefig('../../figs/'+target+'_resid.pdf')
