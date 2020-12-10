import os, sys, time
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from deproject_vis import deproject_vis


# target-specific inputs
disk_name = ['SR4', 'RULup', 'Elias20',
             'Sz129', 'HD143006', 'GWLup',
             'Elias24', 'HD163296', 'AS209']
disk_lbls = ['SR 4', 'RU Lup', 'Elias 20',
             'Sz 129', 'HD 143006', 'GW Lup',
             'Elias 24', 'HD 163296', 'AS 209']
offs = [[-0.060, -0.509], [-0.017, 0.086], [-0.053, -0.490],
        [0.004, 0.004], [-0.006, 0.023], [0.000, 0.001],
        [0.109, -0.383], [-0.0027, 0.0077], [0.002, -0.003]]
incl = [22.0, 18.9, 54.0, 
        31.8, 16.2, 39.0,
        31.0, 46.7, 34.9]
PA = [ 26.0, 124.0, 153.2,
      154.6, 167.0,  37.0,
       45.0, 133.3,  85.8]

# plotting conventions
plt.style.use('default')
plt.rc('font', size=7)
left, right, bottom, top = 0.06, 0.94, 0.055, 0.99
wspace, hspace = 0.24, 0.08
fig = plt.figure(figsize=(7.5, 6.6))
hr = 0.25
hre = 0.15
gs = gridspec.GridSpec(8, 3, width_ratios=(1, 1, 1), 
                       height_ratios=(1, hr, hre, 1, hr, hre, 1, hr))

# visibility class
class Visibility:
    def __init__(self, vis, u, v, wgt):
        self.vis = vis
        self.u = u
        self.v = v
        self.wgt = wgt

# (u,v) bins
uvbins = np.arange(15, 15000, 15)


# loop through profiles
for i in range(len(disk_name)):

    # Load visibility data
    if np.logical_or(disk_name[i] == 'HD143006', disk_name[i] == 'HD163296'):
        dfile = 'data/'+disk_name[i]+'_data_noarc_spavg.vis.npz'
    else:
        dfile = 'data/'+disk_name[i]+'_continuum_spavg_tbin30s.vis.npz'
    vdat = np.load(dfile)
    u, v, vis, wgt = vdat['u'], vdat['v'], vdat['Vis'], vdat['Wgt']
    dvis = Visibility(vis, u, v, wgt)

    # Load visibility model
    mdlv = np.load('fits/'+disk_name[i]+'_frank_uv_fit.npz')['V']
    mvis = Visibility(mdlv, u, v, wgt)

    # Make azimuthally-averaged, deprojected visibility profiles
    dvp = deproject_vis(dvis, uvbins, incl=incl[i], PA=PA[i],
                        offx=offs[i][0], offy=offs[i][1])
    mvp = deproject_vis(mvis, uvbins, incl=incl[i], PA=PA[i],
                        offx=offs[i][0], offy=offs[i][1])

    # -- REALS
    axr = fig.add_subplot(gs[3 * np.floor_divide(i, 3), i % 3])
    axr.plot([0.01, 10], [0, 0], ':k', lw=0.5)
    axr.plot(1e-6 * dvp.rho_uv, 1e3 * dvp.vis_prof.real, 'dimgray', lw=4)
    axr.plot(1e-6 * mvp.rho_uv, 1e3 * mvp.vis_prof.real, 'r', lw=0.5)
    axr.set_xlim([0.015, 10])
    axr.set_xscale('log')
    axr.set_xticklabels([])
    axr.set_ylim([-0.2 * 1e3 * dvp.vis_prof.real.max(), 
                  1.02 * 1e3 * dvp.vis_prof.real.max()])
    axr.text(0.96, 0.9, disk_lbls[i], ha='right', va='center', 
             transform=axr.transAxes)
    
    # -- IMAGS
    axi = fig.add_subplot(gs[3 * np.floor_divide(i, 3) + 1, i % 3])
    axi.plot([0.01, 10], [0, 0], ':k', lw=0.5)
    axi.plot(1e-6 * dvp.rho_uv, 1e3 * dvp.vis_prof.imag, 'dimgray', lw=4)
    axi.plot(1e-6 * mvp.rho_uv, 1e3 * mvp.vis_prof.imag, 'r', lw=0.5)
    axi.set_xlim([0.015, 10])
    axi.set_xscale('log')
    axi.set_xticks([0.1, 1, 10])
    axi.set_xticklabels(['0.1', '1', '10'])
    iscale = hr * 1.22 * 1e3 * dvp.vis_prof.real.max() / 2 
    axi.set_ylim([-iscale, iscale])

    # targeted labeling
    if (i == 6):
        axr.set_ylabel('real  (mJy)', labelpad=5)
        axi.set_ylabel('imag')
        axi.set_xlabel('deprojected baseline (M$\\lambda$)')
        

fig.subplots_adjust(wspace=wspace, hspace=hspace)
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
fig.savefig('figs/vis_profiles.pdf')
fig.clf()
