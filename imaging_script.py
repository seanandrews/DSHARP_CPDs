execfile('reduction_utils.py')

def imaging_script(infile, outfile):

    mask_ra  = '16h25m56.16s'
    mask_dec = '-24.20.48.71'
    fid_mask = 'circle[[%s, %s], %.1farcsec]' % (mask_ra, mask_dec, 0.7)
    scales = [0, 5, 30, 75, 150]

    tclean_wrapper(vis=infile+'.ms', imagename=outfile, 
                   mask=fid_mask, scales=scales,
                   imsize=3000, cellsize='.003arcsec', threshold='0.05mJy',
                   robust=-0.5, uvtaper=['0.035arcsec', '0.01arcsec', '0deg'])
    exportfits(outfile+'.image', outfile+'.fits', overwrite=True)
