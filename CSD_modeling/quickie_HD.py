import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import fits
from astropy.visualization import (AsinhStretch, LinearStretch, ImageNormalize)
sys.path.append('../')
import diskdictionary as disk


# controls
target = 'HD163296'

# constants
c_ = 2.99792e10
k_ = 1.38057e-16

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
dRA_lims, dDEC_lims = [1.05*rout, -1.05*rout], [-1.05*rout, 1.05*rout]

# intensity limits, and stretch
norm = ImageNormalize(vmin=0, vmax=disk.disk[target]['maxTb'], 
                      stretch=AsinhStretch())
cmap = 'inferno'

### Plot the data image
plt.style.use('default')
fig = plt.figure(figsize=(5.0, 5.0))
gs  = gridspec.GridSpec(1, 1)

# image (sky-plane)
ax = fig.add_subplot(gs[0,0])
Tb = (1e-23 * dimg / beam_area) * c_**2 / (2 * k_ * freq**2)
im = ax.imshow(Tb, origin='lower', cmap=cmap, extent=im_bounds, 
               norm=norm, aspect='equal')

# limits, labeling
ax.set_xlim(dRA_lims)
ax.set_ylim(dDEC_lims)
ax.axis('off')

# adjust layout
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
fig.savefig('HD163296.png', dpi=300)
