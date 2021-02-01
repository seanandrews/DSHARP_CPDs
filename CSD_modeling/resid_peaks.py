import os, sys, time
import numpy as np
from astropy.io import fits
sys.path.append('../')
import diskdictionary as disk


targets = ['SR4', 'RULup', 'Elias20', 'Sz129', 'HD143006', 'GWLup',
           'Elias24', 'HD163296', 'AS209']

targets = ['HD143006']


for i in range(len(targets)):

    # load residual images and headers
    if np.logical_or((targets[i] == 'HD143006'), (targets[i] == 'HD163296')):
        hdu = fits.open('data/'+targets[i]+'_resid_symm.JvMcorr.fits')
    else:
        hdu = fits.open('data/'+targets[i]+'_resid.JvMcorr.fits')
    img = 1e6 * np.squeeze(hdu[0].data)    # in uJy/beam
    hd = hdu[0].header

    # Cartesian sky-plane coordinate system
    nx, ny = hd['NAXIS1'], hd['NAXIS2']
    RAo  = 3600 * hd['CDELT1'] * (np.arange(nx) - (hd['CRPIX1'] - 1))
    DECo = 3600 * hd['CDELT2'] * (np.arange(ny) - (hd['CRPIX2'] - 1))
    x_s, y_s = np.meshgrid(RAo - disk.disk[targets[i]]['dx'],
                           DECo - disk.disk[targets[i]]['dy'])

    # Cartesian disk-plane coordinate system
    inclr = np.radians(disk.disk[targets[i]]['incl'])
    PAr = np.radians(disk.disk[targets[i]]['PA'])
    x_d = (x_s * np.cos(PAr) - y_s * np.sin(PAr)) / np.cos(inclr)
    y_d = (x_s * np.sin(PAr) + y_s * np.cos(PAr))

    # Polar disk-plane coordinate system
    r_d = np.sqrt(x_d**2 + y_d**2)
    az_d = np.arctan2(y_d, x_d)


    # Loop through gaps
    rgap = disk.disk[targets[i]]['rgap']
    wgap = disk.disk[targets[i]]['wgap']
    for j in range(len(rgap)):

        # gap boundaries in Polar sky-plane coordinates
        bndi = (r_d <= (rgap[j] - wgap[j]))
        bndo = (r_d >= (rgap[j] + wgap[j]))

        # apply a boolean mask of gap annulus
        mask = np.ones_like(img, dtype='bool')
        mask[np.logical_or(bndi, bndo)] = 0
        gimg, gx_s, gy_s = img[mask], x_s[mask], y_s[mask]
        gr_d, gaz_d = r_d[mask], az_d[mask]

        # isolate peak
        peak = (gimg == gimg.max())
        pk_xs, pk_ys = gx_s[peak], gy_s[peak]
        pk_r, pk_az = gr_d[peak], gaz_d[peak]
        pk_SB = gimg.max()

        # print outputs for Table 4
        name_str = targets[i].ljust(8)
        r_str = str(np.int(np.round(1e3 * pk_r)))
        az_str = str(np.int(np.round(np.degrees(pk_az))))
        SB_str = str(np.int(np.round(pk_SB)))
        rms_str = str(np.int(np.round(np.ma.std(gimg))))
        noise_str = str(np.int(disk.disk[targets[i]]['RMS']))
        disk_mask = np.zeros_like(img, dtype='bool')
        disk_mask[(r_d <= (1.2*disk.disk[targets[i]]['rout']))] = 1
        dimg = img[disk_mask]
        noise_str = str(np.int(np.round(np.ma.std(dimg))))
        print(name_str + '  ' + 'gap'+str(j) + '  ' + r_str + '  ' + \
              az_str + '  ' + SB_str + '  ' + rms_str + '  ' + noise_str)
