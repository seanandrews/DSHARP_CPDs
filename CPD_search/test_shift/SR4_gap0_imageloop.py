import os, sys, time
import numpy as np
execfile('reduction_utils.py')
execfile('JvM_correction_brief.py')
execfile('ImportMS_shiftSR4.py')
sys.path.append('.')
import diskdictionary as disk

# specify target disk and gap
target = 'SR4'
gap_ix = 0
subsuf = '0'

t0 = time.time()

# create a MS from the residual visibilities
resid_suffix = '.test'	
os.system('rm -rf '+target+'_data.'+resid_suffix+'.ms*')
ImportMS_shiftSR4('../data/'+target+'_data.ms', 'SR4_resid.vis', 
                  suffix=resid_suffix, make_resid=False)

# clean image for the residual visibilities (note this is 1024x1024, not
# the default 3000x3000 used for other imaging; done for speed/space)
spec_mask = disk.disk[target]['cmask']
spec_mask = 'circle[[16h25m56.16s, -24.20.48.2], 0.3arcsec]'

im_outfile = target+'_'+resid_suffix
tclean_wrapper(vis='../data/'+target+'_data.'+resid_suffix+'.ms',
               imagename=im_outfile,
               mask=spec_mask,
               scales=disk.disk[target]['cscales'],
               imsize=1024, cellsize='.003arcsec',
               threshold=disk.disk[target]['cthresh'],
               gain=disk.disk[target]['cgain'],
               cycleniter=disk.disk[target]['ccycleniter'],
               robust=disk.disk[target]['crobust'],
               uvtaper=disk.disk[target]['ctaper'])

# perform the JvM correction
eps = do_JvM_correction_and_get_epsilon(im_outfile)

# export the resulting image to a FITS file
exportfits(im_outfile+'.JvMcorr.image', 
           im_outfile+'.JvMcorr.fits', overwrite=True)
               
# clean up
for ext in ['.image', '.model', '.pb', '.psf', '.residual', '.sumwt']:
    os.system('rm -rf '+im_outfile+ext)
os.system('rm -rf ../data/'+target+'_data.'+resid_suffix+'.ms*')

print(time.time()-t0)
