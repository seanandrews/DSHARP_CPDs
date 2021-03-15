import os, sys
import numpy as np
from astropy.io import fits
sys.path.append('../')
import diskdictionary as disk


targets = ['RULup', 'Elias20', 'Sz129', 'GWLup', 'Elias24', 'AS209']

for i in range(len(targets)):

    # beam solid angle
    idir = '../CSD_modeling/data/'
    hdu = fits.open(idir+'deep_'+targets[i]+'_data.JvMcorr.fits')
    hd = hdu[0].header
    hdu.close()
    bmaj, bmin = hd['BMAJ'] * 3600, hd['BMIN'] * 3600
    beam_area = np.pi * bmaj * bmin / (4 * np.log(2))

    # search annulus solid angle
    rgap = disk.disk[targets[i]]['rgap']
    wgap = disk.disk[targets[i]]['wgap']
    for ir in range(len(rgap)):
        search_area = np.pi * ((rgap[ir]+wgap[ir])**2 - (rgap[ir]-wgap[ir])**2)

        print('%10a  gap%1i  %4f' % (targets[i], ir, search_area / beam_area))
        


