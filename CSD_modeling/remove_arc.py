import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
from astropy.io import fits
from astropy.visualization import (AsinhStretch, LinearStretch, ImageNormalize)
from scipy.interpolate import interp1d

def remove_arc(disk, geom, rbounds, azbounds, rout=1.0, vmin=0, vmax=5e-6):

    # geometry estimate
    incl, PA, offRA, offDEC = geom
    inclr, PAr = np.radians(incl), np.radians(PA)

    ### Load the clean model contents and coordinates for plots
    # image and header
    dhdu = fits.open('data/'+disk+'_data0.model.fits')
    dimg, hdr = np.squeeze(dhdu[0].data), dhdu[0].header

    # parse coordinate frame
    nx, ny = hdr['NAXIS1'], hdr['NAXIS2']
    RA = hdr['CDELT1'] * (np.arange(nx) - (hdr['CRPIX1'] - 1))
    DEC = hdr['CDELT2'] * (np.arange(ny) - (hdr['CRPIX2'] - 1))
    RAo, DECo = 3600 * RA, 3600 * DEC
    RAo_shift, DECo_shift = RAo - offRA, DECo - offDEC
    dRA, dDEC = np.meshgrid(RAo_shift, DECo_shift)

    # disk-frame coordinates
    xd = (dRA * np.cos(PAr) - dDEC * np.sin(PAr)) / np.cos(inclr)
    yd = (dRA * np.sin(PAr) + dDEC * np.cos(PAr))
    rd = np.sqrt(xd**2 + yd**2)
    azd = np.degrees(np.arctan2(yd, xd))

    ### Define a mask that isolates the arc
    # mask inside the ring zone that contains the arc
    rri, rro = rbounds[0], rbounds[1]
    cond_in, cond_out = (rd <= rri), (rd >= rro)
    rmask = np.ones_like(dimg, dtype='bool')
    rmask[np.logical_or(cond_in, cond_out)] = 0

    # azimuthal mask
    azmask = np.zeros_like(dimg, dtype='bool')
    cond_az = ((azd >= azbounds[0]) & (azd <= azbounds[1]))
    azmask[cond_az] = 1

    # composite mask
    mask = azmask * rmask


    # compute the median radial profile in the ring, aside from the arc
    nonarc_mask = np.logical_and(rmask, ~azmask)
    rd_na = rd[nonarc_mask]
    img_na = dimg[nonarc_mask]

    rdx = np.linspace(rri, rro, 128)
    drdx = np.mean(np.diff(rdx))
    imgx = np.empty_like(rdx)
    for i in range(len(rdx)):
        inbin = ((rd_na >= rdx[i] - 0.5*drdx) & (rd_na < rdx[i] + 0.5*drdx))
        imgx[i] = np.median(img_na[inbin])
    

    # remove the median profile from the arc-only model
    rimg = dimg.copy()
    rint = interp1d(rdx, imgx, bounds_error=False, fill_value=0)
    rimg = rint(rd)
    arc_img = (dimg - rimg) * mask

    # also make the logical inverse: the arc-less clean model
    arcless_img = dimg - arc_img


    # save these models into .FITS files
    ohdu = fits.open('data/'+disk+'_data0.model.fits')
    ohdu[0].data = np.expand_dims(np.expand_dims(arc_img, 0), 0)
    ohdu.writeto('data/'+disk+'_data_arc.model.fits', overwrite=True)
    ohdu.close()

    ohdu = fits.open('data/'+disk+'_data0.model.fits')
    ohdu[0].data = np.expand_dims(np.expand_dims(arcless_img, 0), 0)
    ohdu.writeto('data/'+disk+'_data_noarc.model.fits', overwrite=True)
    ohdu.close()


    # run the associated CASA script to get the appropriate visibilities
    os.system('casa --nologger --nologfile -c extract_'+disk+'.py')
