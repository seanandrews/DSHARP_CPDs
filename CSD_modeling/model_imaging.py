import os, sys
import numpy as np
execfile('reduction_utils.py')
execfile('JvM_correction.py')
execfile('ImportMS.py')
sys.path.append('../')
import diskdictionary as disk

# read which disk this is about
target = str(np.loadtxt('whichdisk.txt', dtype='str'))

# load model and residual visibilities into MS format
ImportMS('data/'+target+'_continuum_spavg_tbin30s.ms', 
         'fits/'+target+'_frank_uv_fit', suffix='model')

# Perform the imaging
tclean_wrapper(vis='data/'+target+'_continuum_spavg_tbin30s.model.ms',
               imagename='data/'+target+'_model', 
               mask=disk.disk[target]['cmask'], 
               scales=disk.disk[target]['cscales'],
               imsize=3000, cellsize='.003arcsec', 
               threshold=disk.disk[target]['cthresh'],
               gain=disk.disk[target]['cgain'],
               cycleniter=disk.disk[target]['ccycleniter'],
               robust=disk.disk[target]['crobust'],
               uvtaper=disk.disk[target]['ctaper'])

# Perform the JvM correction
eps = do_JvM_correction_and_get_epsilon('data/'+target+'_model')

# Export FITS files of the original + JvM-corrected images
exportfits('data/'+target+'_model.image', 
           'data/'+target+'_model.fits', overwrite=True)
exportfits('data/'+target+'_model.JvMcorr.image',
           'data/'+target+'_model.JvMcorr.fits', overwrite=True)
