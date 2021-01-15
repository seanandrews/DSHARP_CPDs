import os, sys, time
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from deproject_vis import deproject_vis
sys.path.append('../')
import diskdictionary as disk


# target-specific inputs
targets = ['SR4', 'RULup', 'Elias20',
           'Sz129', 'HD143006', 'GWLup',
           'Elias24', 'HD163296', 'AS209']

iticks = [ [-10, -5, 0, 5, 10],
           [-20, 0, 20], 
           [-10, 0, 10],
           [-10, 0, 10],
           [-5, 0, 5],
           [-10, 0, 10],
           [-30, 0, 30],
           [-100, -50, 0, 50, 100],
           [-30, 0, 30] ]

# plotting conventions
plt.style.use('classic')
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
for i in range(len(targets)):

    # Load visibility data
    if np.logical_or(targets[i] == 'HD143006', targets[i] == 'HD163296'):
        dfile = 'data/'+targets[i]+'_data_symm.vis.npz'
    else:
        dfile = 'data/'+targets[i]+'_continuum_spavg_tbin30s.vis.npz'
    vdat = np.load(dfile)
    u, v, vis, wgt = vdat['u'], vdat['v'], vdat['Vis'], vdat['Wgt']
    dvis = Visibility(vis, u, v, wgt)

    # Load visibility model
    if np.logical_or(targets[i] == 'HD143006', targets[i] == 'HD163296'):
        mdlv = np.load('fits/'+targets[i]+'_symm_frank_uv_fit.npz')['V']
    else:
        mdlv = np.load('fits/'+targets[i]+'_frank_uv_fit.npz')['V']
    mvis = Visibility(mdlv, u, v, wgt)

    # Make azimuthally-averaged, deprojected visibility profiles
    dvp = deproject_vis(dvis, uvbins, 
                        incl=disk.disk[targets[i]]['incl'], 
                        PA=disk.disk[targets[i]]['PA'],
                        offx=disk.disk[targets[i]]['dx'], 
                        offy=disk.disk[targets[i]]['dy'])
    mvp = deproject_vis(mvis, uvbins, 
                        incl=disk.disk[targets[i]]['incl'], 
                        PA=disk.disk[targets[i]]['PA'],
                        offx=disk.disk[targets[i]]['dx'], 
                        offy=disk.disk[targets[i]]['dy'])

    # -- REALS
    axr = fig.add_subplot(gs[3 * np.floor_divide(i, 3), i % 3])
    axr.plot([0.01, 10], [0, 0], ':k', lw=0.5)
    axr.plot(1e-6 * dvp.rho_uv, 1e3 * dvp.vis_prof.real, 'darkgray', lw=3)
    axr.plot(1e-6 * mvp.rho_uv, 1e3 * mvp.vis_prof.real, 'r')
    axr.set_xlim([0.015, 10])
    axr.set_xscale('log')
    axr.set_xticklabels([])
    axr.set_ylim([-0.2 * 1e3 * dvp.vis_prof.real.max(), 
                  1.02 * 1e3 * dvp.vis_prof.real.max()])
    axr.text(0.94, 0.89, disk.disk[targets[i]]['label'], ha='right', 
             va='center', transform=axr.transAxes)
    
    # -- IMAGS
    axi = fig.add_subplot(gs[3 * np.floor_divide(i, 3) + 1, i % 3])
    axi.plot([0.01, 10], [0, 0], ':k', lw=0.5)
    axi.plot(1e-6 * dvp.rho_uv, 1e3 * dvp.vis_prof.imag, 'darkgray', lw=3)
    axi.plot(1e-6 * mvp.rho_uv, 1e3 * mvp.vis_prof.imag, 'r')
    axi.set_xlim([0.015, 10])
    axi.set_xscale('log')
    axi.set_xticks([0.1, 1, 10])
    axi.set_xticklabels(['0.1', '1', '10'])
    iscale = hr * 1.22 * 1e3 * dvp.vis_prof.real.max() / 2 
    axi.set_ylim([-iscale, iscale])
    axi.set_yticks(iticks[i])

    # targeted labeling
    if (i == 6):
        axr.set_ylabel('real  (mJy)', labelpad=5)
        axi.set_ylabel('imag')
        axi.set_xlabel('deprojected baseline (M$\\lambda$)')
        

fig.subplots_adjust(wspace=wspace, hspace=hspace)
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
fig.savefig('../figs/vis_profiles.pdf')
fig.clf()
