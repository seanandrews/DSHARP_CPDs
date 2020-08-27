execfile('reduction_utils.py')

def HD143006_imaging_script(infile, outfile):

    mask_ra  = '15h58m36.9s'
    mask_dec = '-22.57.15.60'
    mask_rad = 0.8
    fid_mask = 'circle[[%s, %s], %.1farcsec]' % \
               (mask_ra, mask_dec, mask_rad)
    scales = [0, 5, 30, 75]

    tclean_wrapper(vis=infile+'.ms', imagename=outfile, 
                   mask=fid_mask, scales=scales,
                   imsize=3000, cellsize='.003arcsec', threshold='0.05mJy',
                   robust=0., uvtaper=['0.042arcsec', '0.02arcsec', '172.1deg'])
    exportfits(outfile+'.image', outfile+'.fits', overwrite=True)
