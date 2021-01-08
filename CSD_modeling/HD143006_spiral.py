import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from astropy.visualization import (LinearStretch, AsinhStretch, ImageNormalize)
from razmap import razmap
sys.path.append('../')
import diskdictionary as disk

# set color maps
c2 = plt.cm.Reds(np.linspace(0, 1, 32))
c1 = plt.cm.Blues_r(np.linspace(0, 1, 32))
c1 = np.vstack([c1, np.ones((12, 4))])
colors = np.vstack((c1, c2))
dmap = mcolors.LinearSegmentedColormap.from_list('eddymap', colors)
vspan = 10
cmap = 'inferno'


# plotting conventions
rout = disk.disk['HD143006']['rout']
rs = 1.2
plt.style.use('classic')
plt.rc('font', size=6)
left, right, bottom, top = 0.10, 0.88, 0.15, 0.99
wspace, hspace = 0.33, 0.05
xyticks = [-0.5, 0.0, 0.5]
rticks = [0.0, 0.2, 0.4, 0.6]
fig = plt.figure(figsize=(3.5, 1.9))
gs = gridspec.GridSpec(2, 2, height_ratios=(1, 1), width_ratios=(0.5, 1))

# set cylindrical coordinate grids
rbins = np.linspace(0.000, rout*rs, 201)
tbins = np.linspace(-180, 180, 181)


# spiral model parameters
R0 = 30. / disk.disk['HD143006']['distance']
b = 0.3




# Identify image file
dfile = 'data/HD143006_resid_symm.JvMcorr.fits'

# Cylindrical deprojection
rt = razmap(dfile, rbins, tbins, incl=disk.disk['HD143006']['incl'],
            PA=disk.disk['HD143006']['PA'], offx=disk.disk['HD143006']['dx'], 
            offy=disk.disk['HD143006']['dy'])

# Image setups
im_bounds = (rt.dRA.max(), rt.dRA.min(), rt.dDEC.min(), rt.dDEC.max())
dRA_lims, dDEC_lims = [rs*rout, -rs*rout], [-rs*rout, rs*rout]
rt_bounds = (rbins.min(), rbins.max(), tbins.min(), tbins.max())
rlims, tlims = [0, rbins.max()], [tbins.min(), tbins.max()]

# intensity limits and stretch
dnorm = ImageNormalize(vmin=-vspan, vmax=vspan, stretch=LinearStretch())
cnorm = ImageNormalize(vmin=0, vmax=200, stretch=AsinhStretch())


# Diverging residual S/N map (sky-plane, Cartesian)

# set location in figure
ax = fig.add_subplot(gs[0,0])

# plot the residual S/N image
im = ax.imshow(1e6 * rt.image / disk.disk['HD143006']['RMS'], origin='lower',
               cmap=dmap, extent=im_bounds, norm=dnorm, aspect='auto')

# clean beam
beam = Ellipse((dRA_lims[0] + 0.1*np.diff(dRA_lims),
                dDEC_lims[0] + 0.1*np.diff(dDEC_lims)), 
               rt.bmaj, rt.bmin, 90-rt.bpa)
beam.set_facecolor('k')
ax.add_artist(beam)

# limits and labeling
ax.set_xlim(dRA_lims)
ax.set_ylim(dDEC_lims)
ax.set_xticks(xyticks[::-1])
ax.set_yticks(xyticks)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.tick_params(axis='y', length=2.5)
ax.tick_params(axis='x', length=2.0)


# Diverging residual S/N map (disk-plane, Polar)

# set location in figure
ax = fig.add_subplot(gs[0,1])

# plot the residual SNR polar deprojection map
im = ax.imshow(rt.raz_map * 1e6 / disk.disk['HD143006']['RMS'], origin='lower', 
               cmap=dmap, extent=rt_bounds, norm=dnorm, aspect='auto')

# limits and labeling
ax.set_xlim(rlims)
ax.set_ylim(tlims)
ax.set_xticks(rticks)
ax.set_yticks([-180, -90, 0, 90, 180])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.tick_params(axis='y', length=2.0)
ax.tick_params(axis='x', length=2.0)

# colorbar
cbax = fig.add_axes([right + 0.01, 0.59, 0.015, 0.39])
cb = Colorbar(ax=cbax, mappable=im, orientation='vertical', 
              ticklocation='right', ticks=[-9, -6, -3, 0, 3, 6, 9])
cbax.tick_params('both', length=2, direction='out', which='major')
cb.set_label('residual S/N', rotation=270, labelpad=9)



# Convolved residual emission map (sky-plane, Cartesian)
plt.style.use('default')
plt.rc('font', size=6)

# set location in figure
ax = fig.add_subplot(gs[1,0])

# plot the residual image
im = ax.imshow(1e6 * rt.image, origin='lower', cmap=cmap, extent=im_bounds, 
               norm=cnorm, aspect='auto')

# clean beam
beam = Ellipse((dRA_lims[0] + 0.1*np.diff(dRA_lims),
                dDEC_lims[0] + 0.1*np.diff(dDEC_lims)),
               rt.bmaj, rt.bmin, 90-rt.bpa)
beam.set_facecolor('w')
ax.add_artist(beam)

# limits and labeling
ax.set_xlim(dRA_lims)
ax.set_ylim(dDEC_lims)
ax.set_xticks(xyticks[::-1])
ax.set_yticks(xyticks)
ax.set_xlabel('RA offset ($^{\prime\prime}$)', labelpad=3)
ax.set_ylabel('DEC offset ($^{\prime\prime}$)', labelpad=-2)
ax.tick_params(axis='y', length=2.5)
ax.tick_params(axis='x', length=2.5)


# Convolved residual emission map (disk-plane, Polar)

# set location in figure
ax = fig.add_subplot(gs[1,1])

# plot the residual polar deprojection map
im = ax.imshow(rt.raz_map * 1e6, origin='lower', cmap=cmap, extent=rt_bounds,
               norm=cnorm, aspect='auto')

# overplot the spiral model
Rsp = R0 * np.exp(b * np.radians(tbins))
ax.plot(Rsp, tbins, ':w', lw=1, alpha=0.75)

# limits and labeling
ax.set_xlim(rlims)
ax.set_ylim(tlims)
ax.set_yticks([-180, -90, 0, 90, 180])
ax.set_xlabel('radius ($^{\prime\prime}$)', labelpad=3)
ax.set_ylabel('azimuth ($^\circ$)', labelpad=-5)
ax.tick_params(axis='y', length=2.5)
ax.tick_params(axis='x', length=2.5)

# colorbar
cbax = fig.add_axes([right + 0.01, 0.16, 0.015, 0.39])
cb = Colorbar(ax=cbax, mappable=im, orientation='vertical', 
              ticklocation='right')
cbax.tick_params('both', length=2, direction='out', which='major')
cb.set_label('residual ($\\mu$Jy / bm)', rotation=270, labelpad=7)



# Configure plots
fig.subplots_adjust(wspace=wspace, hspace=hspace)
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
fig.savefig('../figs/HD143006_spiral.pdf')
fig.clf()
