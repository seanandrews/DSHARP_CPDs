import os, sys
import numpy as np
from astropy.io import fits
sys.path.append('../')
import diskdictionary as disk

def custom_mask(name, gix, filecopy, buffer_factor=1.0):

    # load the initial mask file, to be used as basis for gap masks
    hdu = fits.open(filecopy+'.mask.fits')
    im_dim, im = np.shape(hdu[0].data), np.squeeze(hdu[0].data)
    hd = hdu[0].header
    hdu.close()

    # Cartesian sky-plane coordinate system
    xin = 3600 * hd['CDELT1'] * (np.arange(hd['NAXIS1']) - (hd['CRPIX1'] - 1))
    yin = 3600 * hd['CDELT2'] * (np.arange(hd['NAXIS2']) - (hd['CRPIX2'] - 1))
    xs, ys = np.meshgrid(xin - disk.disk[name]['dx'], 
                         yin - disk.disk[name]['dy'])

    # Cartesian and polar disk-plane coordinate system
    inclr = np.radians(disk.disk[name]['incl'])
    PAr = np.radians(disk.disk[name]['PA'])
    xd = (xs * np.cos(PAr) - ys * np.sin(PAr)) / np.cos(inclr)
    yd = (xs * np.sin(PAr) + ys * np.cos(PAr))
    rd, azd = np.sqrt(xd**2 + yd**2), np.arctan2(yd, xd)

    # a baseline boolean mask image
    mask = np.zeros_like(im) 

    # identify gap parameters
    rgap, wgap = disk.disk[name]['rgap'][gix], disk.disk[name]['wgap'][gix]

    # gap boundaries in polar disk-plane coordinates
    bndi = (rd >= (rgap - buffer_factor * wgap))
    bndo = (rd <= (rgap + buffer_factor * wgap))

    # update the mask for this gap
    mask[np.logical_and(bndi, bndo)] = 1

    # pack the mask back into a FITS file
    ohdu = fits.open(filecopy+'.mask.fits')
    ohdu[0].data = np.expand_dims(np.expand_dims(mask, 0), 0)
    ohdu.writeto(filecopy+'.custom_mask.fits', overwrite=True)
    ohdu.close()

    return 0
