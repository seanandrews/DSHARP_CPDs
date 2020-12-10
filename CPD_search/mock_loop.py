import os, sys, time
import numpy as np
from inject_CPD import inject_CPD
from frank.geometry import FixedGeometry
from frank.radial_fitters import FrankFitter
from frank.io import save_fit


# specify datafile
dfile = 'SR4_continuum'

# fixed geometric parameters of disk
incl, PA, offRA, offDEC = 22., 26., -0.060, -0.509

# set up mock loops
F_cpd = np.arange(0.25, 0.09, -0.01)	# in mJy
r_cpd = 0.08
n_random_az = 100  			# number of mocks / F_cpd


# set fixed geometry object for Frankenstein
geom = FixedGeometry(incl, PA, dRA=offRA, dDec=offDEC)

# set Frankenstein hyperparameters
Rmax, Ncoll, alpha, wsmth = 0.5, 300, 1.3, 0.1

# set up Frankenstein fitter class
FF = FrankFitter(Rmax=Rmax, N=Ncoll, geometry=geom, alpha=alpha, 
                 weights_smooth=wsmth)


# for safety, load a copy of the dataset
os.system('rm -rf data.vis.npz')
os.system('cp data/'+dfile+'.vis.npz data.vis.npz')
dat = np.load('data.vis.npz')
u, v, vis, wgt = dat['u'], dat['v'], dat['Vis'], dat['Wgt']
data = u, v, vis, wgt

# loop through mock injection and modeling 
os.system('rm -rf '+dfile+'_mpars.txt')
for i in range(len(F_cpd)):

    # assign random azimuths
    az_cpd = np.random.randint(-180, 180, n_random_az)

    for j in range(n_random_az):

        # bookkeeping
        file_suffix = '_F'+str(np.int(1e3*F_cpd[i]))+'uJy_'+str(j).zfill(4)
        print('F'+str(np.int(1e3*F_cpd[i]))+'    '+str(j).zfill(4))

        # inject a mock CPD into the data
        CPD_pars = F_cpd[i], r_cpd, az_cpd[j]
        dvis_wcpd = inject_CPD(data, CPD_pars, incl=incl, PA=PA, 
                               offRA=offRA, offDEC=offDEC)

        # Frankenstein modeling of the data + CPD injection
        sol = FF.fit(u, v, dvis_wcpd, wgt)

        # save the results
        save_fit(u, v, dvis_wcpd, wgt, sol, prefix=dfile+file_suffix,
                 save_vis_fit=False, save_solution=False)

        # clean up file outputs
        os.system('mv '+dfile+file_suffix+'_frank_uv_resid.npz resid_vis/')
        os.system('mv '+dfile+file_suffix+'_frank_profile_fit.txt mprofiles/')
        #os.system('mv '+dfile+file_suffix+'_frank_uv_fit.npz model_vis/')
        os.system('rm '+dfile+file_suffix+'_frank*')

        # record parameter values (F_cpd in uJy, j, r_cpd, az_cpd)
        with open(dfile+'_mpars.txt', 'a') as f:
            f.write('%i    %s    %.3f    %i\n' % \
                    (np.int(1e3*F_cpd[i]), str(j).zfill(4), r_cpd, az_cpd[j]))
