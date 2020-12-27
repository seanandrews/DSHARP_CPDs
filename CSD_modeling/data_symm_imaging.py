import os, sys
import numpy as np
execfile('reduction_utils.py')
execfile('JvM_correction.py')
execfile('ImportMS.py')
sys.path.append('../')
import diskdictionary as disk

# read which disk this is about
target = str(np.loadtxt('whichdisk.txt', dtype='str'))

# Perform the imaging
tclean_wrapper(vis='data/'+target+'_data_symm.ms',
               imagename='data/'+target+'_data_symm', 
               mask=disk.disk[target]['cmask'], 
               scales=disk.disk[target]['cscales'],
               imsize=3000, cellsize='.003arcsec', 
               threshold=disk.disk[target]['cthresh'],
               gain=disk.disk[target]['cgain'],
               cycleniter=disk.disk[target]['ccycleniter'],
               robust=disk.disk[target]['crobust'],
               uvtaper=disk.disk[target]['ctaper'])

# Perform the JvM correction
eps = do_JvM_correction_and_get_epsilon('data/'+target+'_data_symm')

# Estimate map RMS as in DSHARP
coords = str.split(str.split(disk.disk[target]['cmask'], ']')[0], '[[')[1]
noise_ann = "annulus[[%s], ['%.2farcsec', '4.25arcsec']]" % \
            (coords, 1.2 * disk.disk[target]['rout'])
estimate_SNR('data/'+target+'_data_symm.JvMcorr.image', 
             disk_mask = disk.disk[target]['cmask'], noise_mask=noise_ann)
print('epsilon = ', eps)

# Export FITS files of the original + JvM-corrected images
exportfits('data/'+target+'_data_symm.image', 
           'data/'+target+'_data_symm.fits', overwrite=True)
exportfits('data/'+target+'_data_symm.JvMcorr.image',
           'data/'+target+'_data_symm.JvMcorr.fits', overwrite=True)
