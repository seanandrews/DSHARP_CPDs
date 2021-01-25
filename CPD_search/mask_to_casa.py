import os, sys
import numpy as np

target, gap_ix, subsuf = np.loadtxt('whichdisk.txt', dtype=str)

importfits(target+'_gap'+gap_ix+'.'+subsuf+'.custom_mask.fits',
           target+'_gap'+gap_ix+'.'+subsuf+'.custom.mask', overwrite=True)
