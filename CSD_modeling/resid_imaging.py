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
         'fits/'+target+'_frank_uv_resid', suffix='resid')

# Perform the imaging
imagename = 'data/deep_'+target+'_resid'
for ext in ['.image', '.mask', '.model', '.pb', '.psf', '.residual', '.sumwt']:
    os.system('rm -rf '+imagename+ext)
tclean(vis='data/'+target+'_continuum_spavg_tbin30s.resid.ms', 
       imagename=imagename, specmode='mfs', deconvolver='multiscale',
       scales=disk.disk[target]['cscales'], mask=disk.disk[target]['cmask'], 
       imsize=1024, cell='.006arcsec', gain=disk.disk[target]['cgain'],
       cycleniter=disk.disk[target]['ccycleniter'], cyclefactor=1, nterms=1,
       weighting='briggs', robust=disk.disk[target]['crobust'],
       uvtaper=disk.disk[target]['ctaper'], 
       niter=50000, threshold=disk.disk[target]['gthresh'], savemodel='none')

# Perform the JvM correction
eps = do_JvM_correction_and_get_epsilon(imagename)

# Estimate map RMS as in DSHARP
coords = str.split(str.split(disk.disk[target]['cmask'], ']')[0], '[[')[1]
noise_ann = "annulus[[%s], ['%.2farcsec', '4.5arcsec']]" % \
            (coords, 1.2 * disk.disk[target]['rout'])
estimate_SNR(imagename+'.JvMcorr.image',
             disk_mask=disk.disk[target]['cmask'], noise_mask=noise_ann)

# Export FITS files of the original + JvM-corrected images
exportfits(imagename+'.image', imagename+'.fits', overwrite=True)
exportfits(imagename+'.JvMcorr.image', imagename+'.JvMcorr.fits', 
           overwrite=True)
