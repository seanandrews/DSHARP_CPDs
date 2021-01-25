import os, sys, time
import numpy as np
execfile('JvM_correction_brief.py')
execfile('ImportMS.py')
sys.path.append('../')
import diskdictionary as disk

# specify target disk and gap
target, gap_ix, subsuf = np.loadtxt('whichdisk.txt', dtype=str)

# load mock injection parameters file data (as strings)
inj_file = 'injections/'+target+'_gap'+gap_ix+'_mpars.'+subsuf+'.txt'
Fstr, mstr, rstr, azstr = np.loadtxt(inj_file, dtype=str).T


t0 = time.time()

# create a MS from the residual visibilities
rfile = 'resid_vis/' + target + '_gap' + gap_ix + \
        '_F' + Fstr[0] + 'uJy_' + mstr[0] + '_frank_uv_resid'
resid_suffix = 'gap'+gap_ix+'.F'+Fstr[0]+'uJy_'+mstr[0]+'.resid'
os.system('rm -rf '+target+'_data.'+resid_suffix+'.ms*')
ImportMS('data/'+target+'_data.ms', rfile, suffix=resid_suffix, 
         make_resid=False)

# make a set of images that can be used as a basis for imageloop
im_outfile = target+'_'+resid_suffix
for ext in ['.image', '.mask', '.model', '.pb', '.psf', '.residual', '.sumwt']:
    os.system('rm -rf '+im_outfile+ext)
tclean(vis='data/'+target+'_data.'+resid_suffix+'.ms', imagename=im_outfile,
       specmode='mfs', deconvolver='multiscale', imsize=1024, cell='.006arcsec',
       scales=disk.disk[target]['cscales'], mask=disk.disk[target]['cmask'], 
       gain=0.3, cycleniter=300, cyclefactor=1, nterms=1, niter=50000,
       weighting='briggs', robust=disk.disk[target]['crobust'], 
       uvtaper=disk.disk[target]['ctaper'], savemodel='none', 
       threshold=disk.disk[target]['cthresh'], interactive=False)

# perform the JvM correction
eps = do_JvM_correction_and_get_epsilon(im_outfile)

# export the mask to a FITS file
exportfits(im_outfile+'.mask', 
           target+'_gap'+gap_ix+'.'+subsuf+'.mask.fits', overwrite=True)
               
# clean up
for ext in ['.image', '.mask', '.model', '.pb', '.residual', '.JvMcorr.image']:
    os.system('rm -rf '+im_outfile+ext)
os.system('rm -rf data/'+target+'_data.'+resid_suffix+'.ms*')
os.system('rm -rf '+target+'_gap'+gap_ix+'.'+subsuf+'.psf')
os.system('mv '+im_outfile+'.psf '+target+'_gap'+gap_ix+'.'+subsuf+'.psf')
os.system('rm -rf '+target+'_gap'+gap_ix+'.'+subsuf+'.sumwt')
os.system('mv '+im_outfile+'.sumwt '+target+'_gap'+gap_ix+'.'+subsuf+'.sumwt')

print(time.time()-t0)
