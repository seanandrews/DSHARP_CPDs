import os
import numpy as np
execfile('ExportMS.py')

# read which disk this is about
target = str(np.loadtxt('whichdisk.txt', dtype='str'))

# load FITS model of the arc (the modified CLEAN model)
importfits(fitsimage='data/'+target+'_data_arc.cleanmodel.fits',
           imagename='data/'+target+'_data_arc.cleanmodel', overwrite=True)

# Fourier transform the arc model onto the same (u,v) tracks as in the data
# MS (make a copy first!), and store them in the 'MODEL_DATA' column
os.system('rm -rf data/'+target+'_data.ms*')
os.system('cp -r data/'+target+'_continuum_spavg_tbin30s.ms data/temp_' + \
          target+'.ms')
ft(vis='data/temp_'+target+'.ms', 
   model='data/'+target+'_data_arc.model', usescratch=True)

# Now subtract the FT of the arc model from the observed visibilities; the 
# result is stored in the 'CORRECTED_DATA' column
uvsub(vis='data/temp_'+target+'.ms')

# Split out the 'CORRECTED_DATA' visibilities into their own "arc-less" MS
os.system('rm -rf data/'+target+'_data_symm.ms*')
split(vis='data/temp_'+target+'.ms', outputvis='data/'+target+'_data_symm.ms', 
      datacolumn='corrected')

# Export the "arc-less" MS into npz format for frankenstein modeling
ExportMS('data/'+target+'_data_symm.ms')

# Clean up
os.system('rm -rf data/temp_'+target+'.ms*')
os.system('rm -rf data/'+target+'_data_symm_spavg.ms*')
os.system('mv data/'+target+'_data_symm_spavg.vis.npz ' + \
          'data/'+target+'_data_symm.vis.npz')
