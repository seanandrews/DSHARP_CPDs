import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
import matplotlib.colors as mcolors
from astropy.visualization import (LinearStretch, ImageNormalize)
from razmap import razmap
sys.path.append('../')
import diskdictionary as disk

# set color map
c2 = plt.cm.Reds(np.linspace(0, 1, 32))
c1 = plt.cm.Blues_r(np.linspace(0, 1, 32))
c1 = np.vstack([c1, np.ones((26, 4))])
colors = np.vstack((c1, c2))
cmap = mcolors.LinearSegmentedColormap.from_list('eddymap', colors)
vspan = 5

# targets
targets = ['GWLup', 'Elias24', 'HD163296', 'AS209']


# plotting conventions
rout = 0.80
rs = 1.5
plt.style.use('classic')
plt.rc('font', size=6)
left, right, bottom, top = 0.10, 0.86, 0.07, 0.985
wspace, hspace = 0.03, 0.03
fig = plt.figure(figsize=(3.5, 4.4))
gs = gridspec.GridSpec(4, 1)

# set cylindrical coordinate grids
rbins = np.linspace(0.003, rout*rs, 200)
tbins = np.linspace(-180, 180, 181)


# target loop (index i)
for i in range(len(targets)):

    # Identify image files
    if (targets[i] == 'HD163296'):
        dfile = 'data/'+targets[i]+'_resid_symm.JvMcorr.fits'
    else:
        dfile = 'data/'+targets[i]+'_resid.JvMcorr.fits'

    # Cylindrical deprojection
    rt = razmap(dfile, rbins, tbins, incl=disk.disk[targets[i]]['incl'],
                PA=disk.disk[targets[i]]['PA'],
                offx=disk.disk[targets[i]]['dx'], 
                offy=disk.disk[targets[i]]['dy'])

    # Image setups
    im_bounds = (rbins.min(), rbins.max(), tbins.min(), tbins.max())
    rlims, tlims = [0, rbins.max()], [tbins.min(), tbins.max()]

    # intensity limits and stretch
    norm = ImageNormalize(vmin=-vspan, vmax=vspan, stretch=LinearStretch())

    # set location in figure
    ax = fig.add_subplot(gs[i,0])

    # plot the residual SNR polar deprojection map
    im = ax.imshow(rt.raz_map * 1e6 / disk.disk[targets[i]]['RMS'], 
                   origin='lower', cmap=cmap, extent=im_bounds, norm=norm,
                   aspect='auto')

    # mark the gaps
    hatch = 'XXX'
    hatchcol = 'darkgray'
    hatchalp = 0.2
    plt.rc('hatch', linewidth=0.5)

    rgap = disk.disk[targets[i]]['rgap']
    wgap = disk.disk[targets[i]]['wgap']
    wbm = np.sqrt(rt.bmaj * rt.bmin) / 2.355
    ax.fill_between([0, rgap[0]-wgap[0]-wbm], [tlims[1], tlims[1]],
                    [tlims[0], tlims[0]], facecolor='none', alpha=hatchalp,
                    hatch=hatch, edgecolor=hatchcol, linewidth=0.0)
    if (len(rgap) == 1):
        ax.fill_between([rgap[0]+wgap[0]+wbm, rlims[1]], [tlims[1], tlims[1]],
                        [tlims[0], tlims[0]], facecolor='none', alpha=hatchalp,
                        hatch=hatch, edgecolor=hatchcol, linewidth=0.0)
    else:
        ax.fill_between([rgap[0]+wgap[0]+wbm, rgap[1]-wgap[1]-wbm],
                        [tlims[1], tlims[1]], [tlims[0], tlims[0]],
                        facecolor='none', alpha=hatchalp, hatch=hatch,
                        edgecolor=hatchcol, linewidth=0.0)
        ax.fill_between([rgap[1]+wgap[1]+wbm, rlims[1]], [tlims[1], tlims[1]],
                        [tlims[0], tlims[0]], facecolor='none', alpha=hatchalp,
                        hatch=hatch, edgecolor=hatchcol, linewidth=0.0)
    for ir in range(len(rgap)):
        ax.plot([rgap[ir] - wgap[ir], rgap[ir] - wgap[ir]],
                tlims, ':k', lw=0.5, alpha=0.5)
        ax.plot([rgap[ir] + wgap[ir], rgap[ir] + wgap[ir]],
                tlims, ':k', lw=0.5, alpha=0.5)

    # limits and labeling
    ax.text(rlims[1] - 0.035*np.diff(rlims), tlims[1] - 0.18*np.diff(tlims),
            disk.disk[targets[i]]['label'], ha='right', color='k', fontsize=6,
            bbox={'facecolor': 'w', 'edgecolor': 'w', 'pad': 2})
    ax.set_xlim(rlims)
    ax.set_ylim(tlims)
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.tick_params(axis='y', length=2)
    ax.tick_params(axis='x', length=2)

    if (i == 3):
        ax.set_xlabel('radius  ($^{\prime\prime}$)', labelpad=2.5)
        ax.set_ylabel('azimuth  ($^{\circ}$)', labelpad=-2)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])


# colorbar
cbax = fig.add_axes([right + 0.02, bottom, 0.025, 0.455])
cb = Colorbar(ax=cbax, mappable=im, orientation='vertical', 
              ticklocation='right')#,
#              ticks=[-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
cbax.tick_params('both', length=2.5, direction='out', which='major')
cb.set_label('residual S/N', rotation=270, labelpad=5)


fig.subplots_adjust(wspace=wspace, hspace=hspace)
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
fig.savefig('../figs/raz2.pdf')
fig.clf()
