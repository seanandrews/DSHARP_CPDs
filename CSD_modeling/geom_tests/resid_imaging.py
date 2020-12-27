import os, sys
import numpy as np
execfile('reduction_utils.py')
execfile('JvM_correction.py')
execfile('ImportMS.py')

# read which disk this is about
target = str(np.loadtxt('whichdisk.txt', dtype='str'))

# load model and residual visibilities into MS format
ImportMS('data/'+target+'.ms', 
         'fits/'+target+'_frank_uv_resid', suffix='resid')

# Perform the imaging
tclean_wrapper(vis='data/'+target+'.resid.ms',
               imagename='data/'+target+'_resid', 
               mask='ellipse[[15h46m44.709s, -34.30.36.076], [1.3arcsec, 1.1arcsec], 110deg]',
               scales=[0, 20, 50, 100, 200],
               imsize=3000, cellsize='.003arcsec', 
               threshold='0.05mJy',
               robust=0.5, uvtaper=['0.035arcsec', '0.015arcsec', '0deg'])

# Perform the JvM correction
eps = do_JvM_correction_and_get_epsilon('data/'+target+'_resid')

# Export FITS files of the original + JvM-corrected images
exportfits('data/'+target+'_resid.image', 
           'data/'+target+'_resid.fits', overwrite=True)
exportfits('data/'+target+'_resid.JvMcorr.image',
           'data/'+target+'_resid.JvMcorr.fits', overwrite=True)
