import os, sys, time
import numpy as np
from astropy.io import fits
sys.path.append('../')
import diskdictionary as disk

# target disk/gap; iteration
target = 'AS209'
gap = 0
ix = '1'

# load the injection file data
inj_file = 'injections/'+target+'_gap'+str(gap)+'_mpars.'+ix+'.txt'
Fstr, mstr, rstr, azstr = np.loadtxt(inj_file, dtype=str).T
Fcpd, mdl, rcpd, azcpd = np.loadtxt(inj_file).T

# bookkeeping
recov_file = 'recoveries/'+target+'_gap'+str(gap)+'_recoveries.'+ix+'.txt'
os.system('rm -rf ' + recov_file)


# loop through injections
for i in range(len(Fstr)):

    # load the residual image, header
    im_file = target + '_gap' + str(gap) + '.F' + Fstr[i] + 'uJy_' + mstr[i]
    hdu = fits.open('/data/sandrews/DSHARP_CPDs/CPD_search/resid_images/' + im_file + '.resid.JvMcorr.fits')
    img = 1e6 * np.squeeze(hdu[0].data)    # in microJy/beam
    hd = hdu[0].header
    hdu.close()

    # Cartesian sky-plane coordinate system
    nx, ny = hd['NAXIS1'], hd['NAXIS2']
    RAo  = 3600 * hd['CDELT1'] * (np.arange(nx) - (hd['CRPIX1'] - 1))
    DECo = 3600 * hd['CDELT2'] * (np.arange(ny) - (hd['CRPIX2'] - 1))
    xs, ys = np.meshgrid(RAo - disk.disk[target]['dx'],
                         DECo - disk.disk[target]['dy'])

    # Cartesian disk-plane coordinate system
    inclr = np.radians(disk.disk[target]['incl'])
    PAr = np.radians(disk.disk[target]['PA'])
    xd = (xs * np.cos(PAr) - ys * np.sin(PAr)) / np.cos(inclr)
    yd = (xs * np.sin(PAr) + ys * np.cos(PAr))

    # Polar disk-plane coordinate system
    rd  = np.sqrt(xd**2 + yd**2)
    azd = np.degrees(np.arctan2(yd, xd))

    # Load gap properties
    rgap = disk.disk[target]['rgap'][gap]
    wgap = disk.disk[target]['wgap'][gap]
    
    # Apply a boolean mask to isolate the search annulus
    mask = np.zeros_like(img, dtype='bool')
    bndi, bndo = (rd >= (rgap - wgap)), (rd <= (rgap + wgap))
    mask[np.logical_and(bndi, bndo)] = 1
    g_img, g_xs, g_ys = img[mask], xs[mask], ys[mask]
    g_rd, g_azd = rd[mask], azd[mask]

    # Locate and measure the peak
    peak = (g_img == g_img.max())
    pk_xs, pk_ys, pk_r, pk_az = g_xs[peak][0], g_ys[peak][0], g_rd[peak][0], g_azd[peak][0]
    pk_SB = g_img.max()

    # Quantify the pixel distribution in the gap region
    dist_peak = np.sqrt((xs-pk_xs)**2 + (ys-pk_ys)**2)
    mask[(dist_peak <= (5 * hd['CDELT2'] * 3600))] = 0
    emean, estd = np.mean(img[mask]), np.std(img[mask])

    # Record the injection / recovery outcomes
    with open(recov_file, 'a') as f:
        f.write('%.0f  %.0f  %s  %.3f  %.3f  %4i  %4i  %.5f  %.5f  %.0f  %.0f\n' % \
                (Fcpd[i], pk_SB, mstr[i], rcpd[i], pk_r, azcpd[i], pk_az,
                 pk_xs, pk_ys, emean, estd))
