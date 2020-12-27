import os, sys, time
import numpy as np
from img_parser import img_parser
from vis_sample import vis_sample

def gen_mdl(pars, name):

    # calculate a model image
    foo = img_parser(inc=pars[0], PA=pars[1], dist=140., r0=10., r_l=pars[2], 
                     z0=pars[3], zpsi=pars[4], zphi=np.inf, 
                     Tb0=pars[5], Tbq=pars[6], Tbeps=np.inf, 
                     Tbmax=500., Tbmax_b=500., FOV=2.56, Npix=512)

    # load the template visibilities
    dat = np.load('data/template.vis.npz')
    u, v, vis, wgt = dat['u'], dat['v'], dat['Vis'], dat['Wgt']

    # FT and sample the model image onto the (u,v) spacings
    mvis = vis_sample(imagefile=foo, uu=u, vv=v, 
                      mu_RA=pars[7], mu_DEC=pars[8], mod_interp=False)
    print(mvis.shape)

    # Save the model and residual visibilities
    os.system('rm -rf data/'+name+'.vis.npz')
    np.savez('data/'+name+'.vis.npz', u=u, v=v, Vis=np.squeeze(mvis), Wgt=wgt)
