import os, sys
import numpy as np
execfile('reduction_utils.py')
execfile('JvM_correction.py')

# read which disk this is about
target = str(np.loadtxt('whichdisk.txt', dtype='str'))

# Perform the imaging
tclean_wrapper(vis='data/'+target+'.ms',
               imagename='data/'+target+'_data', 
               mask='ellipse[[15h46m44.709s, -34.30.36.076], [1.3arcsec, 1.1arcsec], 110deg]',
               scales=[0, 20, 50, 100, 200],
               imsize=3000, cellsize='.003arcsec', threshold='0.05mJy',
               robust=0.5, uvtaper=['0.035arcsec', '0.015arcsec', '0deg'])

# Perform the JvM correction
eps = do_JvM_correction_and_get_epsilon('data/'+target+'_data')

# Export FITS files of the original + JvM-corrected images
exportfits('data/'+target+'_data.image', 
           'data/'+target+'_data.fits', overwrite=True)
exportfits('data/'+target+'_data.JvMcorr.image',
           'data/'+target+'_data.JvMcorr.fits', overwrite=True)
