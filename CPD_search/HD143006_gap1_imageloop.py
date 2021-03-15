import os, sys, time
import numpy as np
execfile('JvM_correction_brief.py')
execfile('ImportMS.py')
sys.path.append('../')
import diskdictionary as disk

# specify target disk and gap
target, gap_ix, subsuf = 'HD143006', '1', '1'


# load mock injection parameters file data (as strings)
inj_file = 'injections/'+target+'_gap'+gap_ix+'_mpars.'+subsuf+'.txt'
Fstr, mstr, rstr, azstr = np.loadtxt(inj_file, dtype=str).T


# loop through each set of residuals and make an image
for i in range(len(Fstr)):

    t0 = time.time()

    # create a MS from the residual visibilities
    rfile = 'resid_vis/'+target+'_gap'+gap_ix+ \
            '_F'+Fstr[i]+'uJy_'+mstr[i]+'_frank_uv_resid'
    resid_suffix = 'gap'+gap_ix+'.F'+Fstr[i]+'uJy_'+mstr[i]+'.resid'
    os.system('rm -rf '+target+'_data.'+resid_suffix+'.ms*')
    ImportMS('data/'+target+'_data.ms', rfile, suffix=resid_suffix, 
             make_resid=False)

    # prepare for imaging
    im_outfile = target+'_'+resid_suffix
    for ext in ['.image', '.model', '.pb', '.residual']:
        os.system('rm -rf '+im_outfile+ext)
    os.system('cp -r '+target+'_gap'+gap_ix+'.'+subsuf+'.psf ' + \
              im_outfile+'.psf')
    os.system('cp -r '+target+'_gap'+gap_ix+'.'+subsuf+'.sumwt ' + \
              im_outfile+'.sumwt')

    # clean
    tclean(vis='data/'+target+'_data.'+resid_suffix+'.ms',
           imagename=im_outfile, specmode='mfs', deconvolver='multiscale',
           imsize=1024, cell='.006arcsec', scales=disk.disk[target]['gscales'],
           mask=target+'_gap'+gap_ix+'.'+subsuf+'.custom.mask', 
           gain=0.3, cycleniter=300, cyclefactor=1, nterms=1, niter=50000,
           weighting='briggs', robust=disk.disk[target]['crobust'],
           uvtaper=disk.disk[target]['ctaper'], savemodel='none',
           threshold=disk.disk[target]['gthresh'], interactive=False, 
           calcpsf=False)

    # perform the JvM correction
    eps = do_JvM_correction_and_get_epsilon(im_outfile)

    # export the resulting images to FITS files
    exportfits(im_outfile+'.JvMcorr.image', 
               'resid_images/'+im_outfile+'.JvMcorr.fits', overwrite=True)
    exportfits(im_outfile+'.image', 'resid_images/'+im_outfile+'.fits',
               overwrite=True)
               
    # clean up
    for ext in ['.image', '.mask', '.model', '.pb', '.psf', '.residual', 
                '.sumwt', '.JvMcorr.image']:
        os.system('rm -rf '+im_outfile+ext)
    os.system('rm -rf data/'+target+'_data.'+resid_suffix+'.ms*')

    print(time.time()-t0)


