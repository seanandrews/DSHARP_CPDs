execfile('reduction_utils.py')

def Elias20_imaging_script(infile, outfile):

    mask_ra  = '16h26m18.87s'
    mask_dec = '-24.28.20.18'
    mask_PA = 154
    mask_maj = 0.8
    mask_min = 0.5
    fid_mask = 'ellipse[[%s, %s], [%.1farcsec, %.1farcsec], %.1fdeg]' % \
               (mask_ra, mask_dec, mask_maj, mask_min, mask_PA)
    scales = [0, 10, 25, 50, 100]

    tclean_wrapper(vis=infile+'.ms', imagename=outfile, 
                   mask=fid_mask, scales=scales,
                   imsize=3000, cellsize='.003arcsec', threshold='0.06mJy',
                   robust=0., gain=0.1, cycleniter=100)
    exportfits(outfile+'.image', outfile+'.fits', overwrite=True)
