execfile('reduction_utils.py')

def Sz129_imaging_script(infile, outfile):

    mask_PA, mask_maj, mask_min = 150, 0.85, 0.7
    fid_mask = 'ellipse[[%s, %s], [%.1farcsec, %.1farcsec], %.1fdeg]' % \
               ('15h59m16.454s', '-41.57.10.693631', mask_maj, mask_min, mask_PA)
    scales = [0, 5, 30, 75]

    tclean_wrapper(vis=infile+'.ms', imagename=outfile, 
                   mask=fid_mask, scales=scales, gain=0.1,
                   imsize=3000, cellsize='.003arcsec', threshold='0.05mJy',
                   robust=0.0)
    exportfits(outfile+'.image', outfile+'.fits', overwrite=True)
