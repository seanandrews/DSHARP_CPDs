import os, sys, time
import numpy as np
from custom_mask import custom_mask
sys.path.append('../')
import diskdictionary as disk

# specify target disk, gap, and mock file index
target, gap_ix, subsuf = 'SR4', '0', '0'

# # # # # # #

# package up information to pass to CASA
f = open('whichdisk.txt', 'w')
f.write(target + '\n' + gap_ix + '\n' + subsuf)
f.close()

# preliminary CASA imaging to set up for image loop
os.system('casa --nogui --nologger --nologfile -c prepimaging_casa.py')

# make a custom mask for the gap of interest
custom_mask(target, np.int(gap_ix), target+'_gap'+gap_ix+'.'+subsuf,
            buffer_factor=2.0)

# make a script to convert custom mask into CASA format
os.system('casa --nogui --nologger --nologfile -c mask_to_casa.py')
