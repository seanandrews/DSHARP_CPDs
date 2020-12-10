import os, sys, time
import numpy as np
from astropy.io import fits
from vis_sample.classes import *
from simple_disk import simple_disk
import matplotlib.pyplot as plt

def img_parser(inc=30., PA=40., x0=0., y0=0., dist=100., 
               r_min=0., r_max=500., r0=10., r_l=100.,
               z0=0., zpsi=1., zphi=np.inf,
               Tb0=50., Tbq=-0.5, Tbeps=np.inf, Tbmax=500., Tbmax_b=20.,
               FOV=None, Npix=128, 
               RA=240., DEC=-40., 
               freq=240e9, datafile=None, outfile=None):


    # constants
    CC = 2.9979245800000e10
    KK = 1.38066e-16
    CC = 2.99792e10


    # generate an emission model
    disk = simple_disk(inc, PA, x0=x0, y0=y0, dist=dist, 
                       r_min=r_min, r_max=r_max, r0=r0, r_l=r_l,
                       z0=z0, zpsi=zpsi, zphi=zphi, Tb0=Tb0, Tbq=Tbq, 
                       Tbeps=Tbeps, Tbmax=Tbmax, Tbmax_b=Tbmax_b,
                       FOV=FOV, Npix=Npix)


    # generate channel maps
    img = disk.get_image()

    # convert from brightness temperatures to Jy / pixel
    pixel_area = (disk.cell_sky * np.pi / (180. * 3600.))**2
    img *= 1e23 * pixel_area * 2 * freq**2 * KK / CC**2


    # if an 'outfile' specified, pack the cube into a FITS file 
    if outfile is not None:
        hdu = fits.PrimaryHDU(img[::-1,:])
        header = hdu.header
    
        # basic header inputs
        header['EPOCH'] = 2000.
        header['EQUINOX'] = 2000.
        header['LATPOLE'] = -1.436915713634E+01
        header['LONPOLE'] = 180.

        # spatial coordinates
        header['CTYPE1'] = 'RA---SIN'
        header['CUNIT1'] = 'DEG'
        header['CDELT1'] = -disk.cell_sky / 3600.
        header['CRPIX1'] = 0.5 * disk.Npix + 0.5
        header['CRVAL1'] = RA
        header['CTYPE2'] = 'DEC--SIN'
        header['CUNIT2'] = 'DEG'
        header['CDELT2'] = disk.cell_sky / 3600.
        header['CRPIX2'] = 0.5 * disk.Npix + 0.5
        header['CRVAL2'] = DEC

        # frequency coordinates
        header['CTYPE3'] = 'FREQ'
        header['CUNIT3'] = 'Hz'
        header['CRPIX3'] = 1.
        header['CDELT3'] = 7.5e9
        header['CRVAL3'] = freq
        header['SPECSYS'] = 'LSRK'
        header['VELREF'] = 257

        # intensity units
        header['BSCALE'] = 1.
        header['BZERO'] = 0.
        header['BUNIT'] = 'JY/PIXEL'
        header['BTYPE'] = 'Intensity'

        # output FITS
        hdu.writeto(outfile, overwrite=True)

        return img[::-1,:]

    # otherwise, return a vis_sample SkyObject
    else:
        # adjust cube formatting
#        mod_data = np.rollaxis(img[:,::-1,:], 0, 3)
        mod_data = np.expand_dims(img[::-1,:], 2)

        # spatial coordinates
        npix_ra = disk.Npix
        mid_pix_ra = 0.5 * disk.Npix + 0.5
        delt_ra = -disk.cell_sky / 3600
        if (delt_ra < 0):
            mod_data = np.fliplr(mod_data)
        mod_ra = (np.arange(npix_ra) - (mid_pix_ra-0.5))*np.abs(delt_ra)*3600
        
        npix_dec = disk.Npix
        mid_pix_dec = 0.5 * disk.Npix + 0.5
        delt_dec = disk.cell_sky / 3600
        if (delt_dec < 0):
            mod_data = np.flipud(mod_data)
        mod_dec = (np.arange(npix_dec)-(mid_pix_dec-0.5))*np.abs(delt_dec)*3600

        # spectral coordinates
        try:
            nchan_freq = len(freqs)
            mid_chan_freq = freqs[0]
            mid_chan = 1
            delt_freq = freqs[1] - freqs[0]
            mod_freqs = (np.arange(nchan_freq)-(mid_chan-1))*delt_freq + \
                        mid_chan_freq
        except:
            mod_freqs = [freq]

        # return a vis_sample SkyImage object
        return SkyImage(mod_data, mod_ra, mod_dec, mod_freqs, None)
