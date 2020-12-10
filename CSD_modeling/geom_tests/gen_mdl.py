import os, sys, time
import numpy as np
from img_parser import img_parser
from vis_sample import vis_sample


# models of interest
mdl_name = 'zr_0.03'
pars = [35., 110., 150., 0.3, 1., 15., 0.5]


# calculate a model image
foo = img_parser(inc=pars[0], PA=pars[1], dist=140., r0=10., r_l=pars[2], 
                 z0=pars[3], zpsi=pars[4], zphi=np.inf, 
                 Tb0=pars[5], Tbq=pars[6], Tbeps=np.inf, 
                 Tbmax=500., Tbmax_b=500., FOV=2.56, Npix=512)

print(np.sum(foo.data))

# load the data visibilities
dat = np.load('fiducial.vis.npz')
u, v, vis, wgt = dat['u'], dat['v'], dat['Vis'], dat['Wgt']

# FT and sample the model image onto the (u,v) spacings
mvis = np.squeeze(vis_sample(imagefile=foo, uu=u, vv=v, 
                             mu_RA=0.0, mu_DEC=0.0, mod_interp=False))

# Save the model and residual visibilities
os.system('rm -rf '+mdl_name+'.model.vis.npz')
np.savez(mdl_name+'.model.vis.npz', u=u, v=v, Vis=mvis, Wgt=wgt)

