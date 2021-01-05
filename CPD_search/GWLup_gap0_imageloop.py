import os, sys, time
import numpy as np
execfile('reduction_utils.py')
execfile('JvM_correction_brief.py')
execfile('ImportMS.py')
sys.path.append('../')
import diskdictionary as disk

# specify target disk and gap
target = 'GWLup'
gap_ix = 0
subsuf = '0'

# load mock injection parameters file data (as strings)
inj_file = target+'_gap'+str(gap_ix)+'_mpars.'+subsuf+'.txt'
Fstr, mstr, rstr, azstr = np.loadtxt(inj_file, dtype=str).T


# loop through each set of residuals and make an image
for i in range(len(Fstr)):

    t0 = time.time()

    # create a MS from the residual visibilities
    rfile = 'resid_vis/' + target + '_gap' + str(gap_ix) + \
            '_F' + Fstr[i] + 'uJy_' + mstr[i] + '_frank_uv_resid'
    resid_suffix = 'gap'+str(gap_ix)+'.F'+Fstr[i]+'uJy_'+mstr[i]+'.resid'
    os.system('rm -rf '+target+'_data.'+resid_suffix+'.ms*')
    ImportMS('data/'+target+'_data.ms', rfile, suffix=resid_suffix, 
             make_resid=False)

    # clean image for the residual visibilities (note this is 1024x1024, not
    # the default 3000x3000 used for other imaging; done for speed/space)
    im_outfile = target+'_'+resid_suffix
    tclean_wrapper(vis='data/'+target+'_data.'+resid_suffix+'.ms',
                   imagename=im_outfile,
                   mask=disk.disk[target]['cmask'],
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
               'resid_images/'+im_outfile+'.JvMcorr.fits', overwrite=True)
               
    # clean up
    for ext in ['.image', '.mask', '.model', '.pb', '.psf', '.residual', 
                '.sumwt', '.JvMcorr.image']:
        os.system('rm -rf '+im_outfile+ext)
    os.system('rm -rf data/'+target+'_data.'+resid_suffix+'.ms*')

    print(time.time()-t0)
