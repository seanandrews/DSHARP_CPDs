# load model and residual visibilities into MS format
execfile('ImportMS.py')
ImportMS('GWLup_continuum_spavg_tbin30s.ms', 'fiducial.vis', 
         suffix='fiducial')

# Perform the imaging
execfile('GWLup_imaging_script.py')
GWLup_imaging_script('GWLup_continuum_spavg_tbin30s.fiducial', 
                     'fiducial') 
