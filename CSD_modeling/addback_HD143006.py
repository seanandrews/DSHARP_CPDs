import os
import numpy as np
execfile('ExportMS.py')

dname = 'HD143006'

# load FITS models
importfits(fitsimage='data/'+dname+'_data_arc.model.fits',
           imagename='data/'+dname+'_data_arc.model', overwrite=True)
#importfits(fitsimage='data/'+dname+'_data_noarc.model.fits',
#           imagename='data/'+dname+'_data_noarc.model', overwrite=True)

# Fourier transform the arc model onto the same (u,v) tracks as in the data
# MS (make a copy first!), and store them in the 'MODEL_DATA' column
os.system('rm -rf data/'+dname+'_data.ms*')
os.system('cp -r data/'+dname+'_continuum_spavg_tbin30s.ms data/' + \
          dname+'_data.ms')
ft(vis='data/'+dname+'_data.ms', 
   model='data/'+dname+'_data_arc.model', usescratch=True)

# Now subtract the FT of the arc model from the observed visibilities; the 
# result is stored in the 'CORRECTED_DATA' column
uvsub(vis='data/'+dname+'_data.ms')

# Split out the 'CORRECTED_DATA' visibilities into their own "arc-less" MS
os.system('rm -rf data/'+dname+'_data_noarc.ms*')
split(vis='data/'+dname+'_data.ms', outputvis='data/'+dname+'_data_noarc.ms', 
      datacolumn='corrected')

# Split out the 'MODEL_DATA' visibilities into their own "arc" MS
#os.system('rm -rf data/'+dname+'_data_arc.ms*')
#split(vis='data/'+dname+'_data.ms', outputvis='data/'+dname+'_data_arc.ms',
#      datacolumn='model')

# Export the "arc-less" MS into npz format for frankenstein modeling
ExportMS('data/'+dname+'_data_noarc.ms')
