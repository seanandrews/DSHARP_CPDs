import os, sys

# read which disk this is about
target = str(np.loadtxt('whichdisk.txt', dtype='str'))

# make a FITS file of the CLEAN model image
exportfits('data/'+target+'_data.model',
           'data/'+target+'_data.cleanmodel.fits', overwrite=True)
