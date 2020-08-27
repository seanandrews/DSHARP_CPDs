execfile('reduction_utils.py')

def RULup_imaging_script(infile, outfile):

    mask_PA, mask_maj, mask_min = 0, 1.2, 1.2
    fid_mask = 'ellipse[[%s, %s], [%.1farcsec, %.1farcsec], %.1fdeg]' % \
               ('15h56m42.29s', '-37.49.15.89', mask_maj, mask_min, mask_PA)
    scales = [0, 5, 30, 75]

    tclean_wrapper(vis=infile+'.ms', imagename=outfile, 
                   mask=fid_mask, scales=scales, 
                   imsize=3000, cellsize='.003arcsec', threshold='0.08mJy',
                   robust=-0.5, uvtaper=['0.022arcsec', '0.01arcsec', '-6deg'])
    exportfits(outfile+'.image', outfile+'.fits', overwrite=True)
