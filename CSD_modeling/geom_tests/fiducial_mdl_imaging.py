# load model and residual visibilities into MS format
execfile('ImportMS.py')
ImportMS('fiducial.ms', 'fiducial.vis', suffix='model')

# Perform the imaging
execfile('GWLup_imaging_script.py')
GWLup_imaging_script('fiducial', 'fiducial.model') 
