import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from deproject_vis import deproject_vis
sys.path.append('../')
import diskdictionary as disk


# controls
target = 'HD143006'

plt.style.use('default')
plt.rc('font', size=21)


# visibility class
class Visibility:
    def __init__(self, vis, u, v, wgt):
        self.vis = vis
        self.u = u
        self.v = v
        self.wgt = wgt

# (u, v) bins
uvbins = np.arange(15, 15000, 15)

# load and package all the visibilities of interest
vdat = np.load('data/'+target+'_continuum_spavg_tbin30s.vis.npz')
u, v, vis, wgt = vdat['u'], vdat['v'], vdat['Vis'], vdat['Wgt']
vis_data = Visibility(vis, u, v, wgt)

vis = np.load('fits/'+target+'_frank_uv_fit.npz')['V']
vis_model = Visibility(vis, u, v, wgt)

vis_resid = Visibility(vis_data.vis - vis_model.vis, u, v, wgt)


vdat = np.load('data/'+target+'_data_symm.vis.npz')
u, v, vis, wgt = vdat['u'], vdat['v'], vdat['Vis'], vdat['Wgt']
vis_data_symm = Visibility(vis, u, v, wgt)

vis = np.load('fits/'+target+'_symm_frank_uv_fit.npz')['V']
vis_model_symm = Visibility(vis, u, v, wgt)

vis_resid_symm = Visibility(vis_data_symm.vis - vis_model_symm.vis, u, v, wgt)


vis_asymm = Visibility(vis_data.vis - vis_data_symm.vis, u, v, wgt)


vispack = [vis_data, vis_model, vis_resid, 
           vis_data_symm, vis_model_symm, vis_resid_symm,
           vis_asymm]

vislbl = ['_vis_data', '_vis_model', '_vis_resid', 
          '_vis_data_symm', '_vis_model_symm', '_vis_resid_symm',
          '_vis_asymm']

vcols = ['dimgray', 'r', 'dimgray',
         'dimgray', 'r', 'dimgray',
         'b']

vmax = 60


for i in range(len(vispack)):

    # Make azimuthally-averaged, deprojected visibility profile
    dvp = deproject_vis(vispack[i], uvbins,
                        incl=disk.disk[target]['incl'],
                        PA=disk.disk[target]['PA'],
                        offx=disk.disk[target]['dx'],
                        offy=disk.disk[target]['dy'])

    ### Set up plot 
    fig = plt.figure(figsize=(7.75, 6.2))
    gs  = gridspec.GridSpec(2, 1, height_ratios=(1, 0.25))

    # Reals
    ax = fig.add_subplot(gs[0,0])
    ax.plot([0.01, 10], [0, 0], ':k', lw=0.5)
    ax.plot(1e-6 * dvp.rho_uv, 1e3 * dvp.vis_prof.real, vcols[i], lw=3)
    ax.set_xlim([0.015, 10])
    ax.set_xscale('log')
    ax.set_xticklabels([])
    ax.set_ylim([-0.2 * vmax, 1.02 * vmax])
    ax.set_ylabel('real  (mJy)', labelpad=21)

    # Imags
    ax = fig.add_subplot(gs[1,0])
    ax.plot([0.01, 10], [0, 0], ':k', lw=0.5)
    ax.plot(1e-6 * dvp.rho_uv, 1e3 * dvp.vis_prof.imag, vcols[i], lw=3)
    ax.set_xlim([0.015, 10])
    ax.set_xscale('log')
    ax.set_xticks([0.1, 1, 10])
    ax.set_xticklabels(['0.1', '1', '10'])
    iscale = 0.25 * 1.22 * vmax / 2
    ax.set_ylim([-iscale, iscale])
    ax.set_ylabel('imag', labelpad=17)
    ax.set_xlabel('deprojected baseline  (M$\\lambda$)')

    # adjust layout
    fig.subplots_adjust(wspace=0.24, hspace=0.07)
    fig.subplots_adjust(left=0.165, right=0.93, bottom=0.125, top=0.985)
    ofile = '../figs/'+target+'_demo'+vislbl[i]
    fig.savefig(ofile+'.pdf')
    fig.clf()
