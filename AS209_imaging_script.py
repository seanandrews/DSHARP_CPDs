execfile('reduction_utils.py')

def AS209_imaging_script(infile, outfile):

    mask_ra  = '16h49m15.29463s'
    mask_dec = '-14.22.09.048165'
    mask_maj = 1.3
    mask_min = 1.1
    mask_pa = 86.
    fid_mask = 'ellipse[[%s, %s], [%.1farcsec, %.1farcsec], %.1fdeg]' % \
               (mask_ra, mask_dec, mask_maj, mask_min, mask_pa)
    scales = [0, 5, 30, 100, 200]

    tclean_wrapper(vis=infile+'.ms', imagename=outfile, 
                   mask=fid_mask, scales=scales, gain=0.2,
                   imsize=3000, cellsize='.003arcsec', threshold='0.08mJy',
                   robust=-0.5, uvtaper=['0.037arcsec', '0.01arcsec', '162deg'])
    exportfits(outfile+'.image', outfile+'.fits', overwrite=True)
