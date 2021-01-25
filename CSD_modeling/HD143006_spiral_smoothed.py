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


# plotting conventions
rout = disk.disk['HD143006']['rout']
rs = 1.35
vmin, vmax = 10, 100

cmap = 'inferno'
plt.style.use('classic')
plt.rc('font', size=6)
left, right, bottom, top = 0.11, 0.89, 0.055, 0.91
wspace, hspace = 0.00, 0.20
xyticks = [-0.5, 0.0, 0.5]
rticks = [0.0, 0.2, 0.4, 0.6]
fig = plt.figure(figsize=(3.5, 5.25))
gs = gridspec.GridSpec(2, 1, height_ratios=(1, 0.5))


# set cylindrical coordinate grids
rbins = np.linspace(0.000, rout*rs, 201)
tbins = np.linspace(-180, 180, 181)


# spiral model parameters
R0 = 30. / disk.disk['HD143006']['distance']
b = 0.3

asp, csp = 0.17, 0.067
tbins_sp = np.linspace(-140, 280, 201)



# Identify image file
dfile = 'data/HD143006_resid_symm.JvMcorr.smoothed.fits'

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
cnorm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())


# Residual emission map (sky-plane, Cartesian)
plt.style.use('default')
plt.rc('font', size=6)

# set location in figure
ax = fig.add_subplot(gs[0,0])

# plot the residual image
im = ax.imshow(1e6 * rt.image, origin='lower', cmap=cmap, extent=im_bounds, 
               norm=cnorm, aspect='equal')

# ring annotations
tt = np.linspace(-np.pi, np.pi, 91)
inclr = np.radians(disk.disk['HD143006']['incl'])
PAr = np.radians(disk.disk['HD143006']['PA'])
rring = [0.036, 0.2473, 0.393]
for ir in range(len(rring)):
    xgi = rring[ir] * np.cos(tt) * np.cos(inclr)
    ygi = rring[ir] * np.sin(tt)
    ax.plot( xgi * np.cos(PAr) + ygi * np.sin(PAr),
            -xgi * np.sin(PAr) + ygi * np.cos(PAr), ':', color='w',
            lw=0.8, alpha=0.7)

# vortex annotations
rbounds, azbounds = [0.393, 0.55], [90., 142.]
tazbins = np.linspace(np.radians(azbounds[0]), np.radians(azbounds[1]), 101)
xai = rbounds[0] * np.cos(tazbins) * np.cos(inclr)
yai = rbounds[0] * np.sin(tazbins)
xao = rbounds[1] * np.cos(tazbins) * np.cos(inclr)
yao = rbounds[1] * np.sin(tazbins)
xarci =  xai * np.cos(PAr) + yai * np.sin(PAr)
yarci = -xai * np.sin(PAr) + yai * np.cos(PAr)
xarco =  xao * np.cos(PAr) + yao * np.sin(PAr)
yarco = -xao * np.sin(PAr) + yao * np.cos(PAr)
ax.plot(xarci, yarci, ':w', lw=0.8, alpha=0.7)
ax.plot(xarco, yarco, ':w', lw=0.8, alpha=0.7)
ax.plot([xarci[0], xarco[0]], [yarci[0], yarco[0]], ':w', lw=0.8, alpha=0.7)
ax.plot([xarci[-1], xarco[-1]], [yarci[-1], yarco[-1]], ':w', lw=0.8, alpha=0.7)

# spiral model annotations
azsp = np.radians(tbins_sp)
Rsp = asp + csp * azsp
xsp = Rsp * np.sin(azsp) * np.sin(PAr) + \
      Rsp * np.cos(azsp) * np.cos(inclr) * np.cos(PAr)
ysp = Rsp * np.sin(azsp) * np.cos(PAr) - \
      Rsp * np.cos(azsp) * np.cos(inclr) * np.sin(PAr)
ax.plot(xsp, ysp, '--c', alpha=0.7)


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
ax.set_ylabel('DEC offset ($^{\prime\prime}$)', labelpad=0)
ax.tick_params(axis='y', length=2.5)
ax.tick_params(axis='x', length=2.5)


# Convolved residual emission map (disk-plane, Polar)

# set location in figure
ax = fig.add_subplot(gs[1,0])

# plot the residual polar deprojection map
im = ax.imshow(rt.raz_map * 1e6, origin='lower', cmap=cmap, extent=rt_bounds,
               norm=cnorm, aspect='auto')

# annotations
for ir in range(len(rring)):
    ax.plot([rring[ir], rring[ir]], [-180, 180], ':w', lw=0.8, alpha=0.7)
ax.plot([rbounds[0], rbounds[1], rbounds[1], rbounds[0], rbounds[0]],
        [azbounds[0], azbounds[0], azbounds[1], azbounds[1], azbounds[0]],
        ':w', lw=0.8, alpha=0.7)

# Archimedean spiral model
ax.plot(Rsp[tbins_sp <= 180], tbins_sp[tbins_sp <= 180], '--c', alpha=0.7)
ax.plot(Rsp[tbins_sp > 180], tbins_sp[tbins_sp > 180] - 360, '--c', alpha=0.7)


for isp in range(len(tbins_sp)):
    print(Rsp[isp], np.degrees(np.arctan2(np.abs(csp), np.abs(Rsp[isp]))))



# limits and labeling
ax.set_xlim(rlims)
ax.set_ylim(tlims)
ax.set_yticks([-180, -90, 0, 90, 180])
ax.set_xlabel('radius ($^{\prime\prime}$)', labelpad=3)
ax.set_ylabel('azimuth ($^\circ$)', labelpad=-1.5)
ax.tick_params(axis='y', length=2.5)
ax.tick_params(axis='x', length=2.5)

# colorbar
cbax = fig.add_axes([left, top + 0.01, right-left, 0.02])
cb = Colorbar(ax=cbax, mappable=im, orientation='horizontal', 
              ticklocation='top')
cbax.tick_params('both', length=2, direction='out', which='major')
cb.set_label('residual ($\\mu$Jy / bm)', labelpad=4)



# Configure plots
fig.subplots_adjust(wspace=wspace, hspace=hspace)
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
fig.savefig('../figs/HD143006_spiral_smoothed.pdf')
fig.clf()
