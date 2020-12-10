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
sys.path.append('../')
import diskdictionary as disk


# controls
target = 'SR4'

frank  = False
im_dat = True
im_res = False
im_mdl = False
annotate_res = True



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
dhdu = fits.open('data/'+target+'_data.JvMcorr.fits')
dimg, hd = np.squeeze(dhdu[0].data), dhdu[0].header

# parse coordinate frame indices into physical numbers
RA = 3600 * hd['CDELT1'] * (np.arange(hd['NAXIS1']) - (hd['CRPIX1'] - 1)) 
DEC = 3600 * hd['CDELT2'] * (np.arange(hd['NAXIS2']) - (hd['CRPIX2'] - 1))
dRA, dDEC = np.meshgrid(RA - disk.disk[target]['dx'], 
                        DEC - disk.disk[target]['dy'])
freq = hd['CRVAL3']

# disk-frame polar coordinates
inclr = np.radians(disk.disk[target]['incl'])
PAr = np.radians(disk.disk[target]['PA'])
xd = (dRA * np.cos(PAr) - dDEC * np.sin(PAr)) / np.cos(inclr)
yd = (dRA * np.sin(PAr) + dDEC * np.cos(PAr))
r, theta = np.sqrt(xd**2 + yd**2), np.degrees(np.arctan2(yd, xd))

# beam parameters
bmaj, bmin, bPA = 3600 * hd['BMAJ'], 3600 * hd['BMIN'], hd['BPA']
beam_area = (np.pi * bmaj * bmin / (4 * np.log(2))) / (3600 * 180 / np.pi)**2

# image setups
rout = disk.disk[target]['rout']
im_bounds = (dRA.max(), dRA.min(), dDEC.min(), dDEC.max())
dRA_lims, dDEC_lims = [1.5*rout, -1.5*rout], [-1.5*rout, 1.5*rout]

# intensity limits, and stretch
norm = ImageNormalize(vmin=0, vmax=disk.disk[target]['maxTb'], 
                      stretch=AsinhStretch())
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

for ir in range(len(disk.disk[target]['rgapi'])):
    rgi = disk.disk[target]['rgapi'][ir]
    rgo = disk.disk[target]['rgapo'][ir]
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
fig.savefig('../figs/'+target+'_dataimage.pdf')




### - AZIMUTHALLY-AVERAGED PROFILE
# compute the profile
rbins = np.arange(hd['CDELT2'] * 3600, 1.5*rout, hd['CDELT2'] * 3600)
dr = np.abs(np.mean(np.diff(rbins)))
SBr, err_SBr = np.empty(len(rbins)), np.empty(len(rbins))
for i in range(len(rbins)):
    in_annulus = ((r >= rbins[i] - 0.5 * dr) & (r < (rbins[i] + 0.5 * dr)))
    SBr[i], err_SBr[i] = np.average(Tb[in_annulus]), np.std(Tb[in_annulus])

# basic plot
fig, ax = plt.subplots(figsize=(7.0, 4.5))
ax.plot(rbins, SBr)

# annotations
for ir in range(len(disk.disk[target]['rgapi'])):
    ri, ro = disk.disk[target]['rgapi'][ir], disk.disk[target]['rgapo'][ir]
    ax.plot([ri, ri], [0.2, 200], ':k')
    ax.plot([ro, ro], [0.2, 200], ':k')
ax.plot([rout, rout], [0.2, 200], '--k')

# labeling
ax.set_xlim(0, 1.5*rout)
ax.set_ylim(0.2, 200)
ax.set_yscale('log')
ax.set_yticks([1, 10, 100])
ax.set_yticklabels(['1', '10', '100'])
ax.set_xlabel('radius  ($^{\prime\prime}$)')
ax.set_ylabel('brightness temperature  (K)')
fig.savefig('../figs/'+target+'_SBr.pdf')



### - FRANK VISIBILITY MODELING 
if frank:
    print('....')
    print('Performing visibility modeling')
    print('....')
    # load the visibility data
    dat = np.load('data/'+target+'_continuum_spavg_tbin30s.vis.npz')
    u, v, vis, wgt = dat['u'], dat['v'], dat['Vis'], dat['Wgt']

    # set the disk viewing geometry
    geom = FixedGeometry(disk.disk[target]['incl'], disk.disk[target]['PA'], 
                         dRA=disk.disk[target]['dx'], 
                         dDec=disk.disk[target]['dy'])

    # configure the fitting code setup
    FF = FrankFitter(Rmax=2*disk.disk[target]['rout'], geometry=geom, 
                     N=disk.disk[target]['hyp-Ncoll'], 
                     alpha=disk.disk[target]['hyp-alpha'], 
                     weights_smooth=disk.disk[target]['hyp-wsmth'])

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
    vmin, vmax = -200, 200    # these are in microJy/beam units
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())
    im = ax.imshow(1e6*rimg, origin='lower', cmap=mymap, extent=im_bounds, 
                   norm=norm, aspect='equal')

    # gap markers
    gcols = ['k', 'darkgray']
    for ir in range(len(disk.disk[target]['rgapi'])):
        rgi = disk.disk[target]['rgapi'][ir]
        rgo = disk.disk[target]['rgapo'][ir]
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
    if annotate_res:
        ax.text(0.05, 0.93, '%.4f,  %.4f,  %.1f,  %.1f' % \
                (disk.disk[target]['dx'], disk.disk[target]['dy'], 
                 disk.disk[target]['incl'], disk.disk[target]['PA']), 
                transform=ax.transAxes)

    # add a scalebar
    cbax = fig.add_subplot(gs[:,1])
    cb = Colorbar(ax=cbax, mappable=im, orientation='vertical',
                  ticklocation='right') 
    cb.set_label('residual brightness  ($\mu$Jy / beam)', rotation=270, 
                 labelpad=18)

    # adjust layout
    fig.subplots_adjust(wspace=0.02)
    fig.subplots_adjust(left=0.11, right=0.89, bottom=0.1, top=0.98)
    fig.savefig('../figs/'+target+'_resid.pdf')
