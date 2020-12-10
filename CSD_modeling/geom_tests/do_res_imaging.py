# load model and residual visibilities into MS format
execfile('ImportMS.py')
ImportMS('fiducial.ms', 'zr_0.03.model.vis', suffix='zr_0.03', make_resid=True)

# Perform the imaging
execfile('fiducial_imaging_script.py')
fiducial_imaging_script('fiducial.zr_0.03.resid', 'data/fiducial_zr_0.03_resid') 
