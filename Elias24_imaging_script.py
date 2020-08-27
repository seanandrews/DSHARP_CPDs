execfile('reduction_utils.py')

def Elias24_imaging_script(infile, outfile):

    mask_ra  = '16h26m24.078s'
    mask_dec = '-24.16.13.883'
    mask_PA = 45
    mask_maj = 1.6
    mask_min = 1.4
    fid_mask = 'ellipse[[%s, %s], [%.1farcsec, %.1farcsec], %.1fdeg]' % \
               (mask_ra, mask_dec, mask_maj, mask_min, mask_PA)
    scales = [0, 20, 50, 100, 200]

    tclean_wrapper(vis=infile+'.ms', imagename=outfile, 
                   mask=fid_mask, scales=scales,
                   imsize=3000, cellsize='.003arcsec', threshold='0.08mJy',
                   robust=0., uvtaper=['0.035arcsec', '0.01arcsec', '166deg'])
    exportfits(outfile+'.image', outfile+'.fits', overwrite=True)
