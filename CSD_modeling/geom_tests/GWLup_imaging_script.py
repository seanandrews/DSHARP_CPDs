execfile('reduction_utils.py')

def GWLup_imaging_script(infile, outfile):

    mask_ra  = '15h46m44.709s'
    mask_dec = '-34.30.36.076'
    mask_PA = 110
    mask_maj = 1.3
    mask_min = 1.1
    fid_mask = 'ellipse[[%s, %s], [%.1farcsec, %.1farcsec], %.1fdeg]' % \
               (mask_ra, mask_dec, mask_maj, mask_min, mask_PA)
    scales = [0, 20, 50, 100, 200]

    tclean_wrapper(vis=infile+'.ms', imagename=outfile, 
                   mask=fid_mask, scales=scales,
                   imsize=3000, cellsize='.003arcsec', threshold='0.05mJy',
                   robust=0.5, uvtaper=['0.035arcsec', '0.015arcsec', '0deg'])
    exportfits(outfile+'.image', outfile+'.fits', overwrite=True)
