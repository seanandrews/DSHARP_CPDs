import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib as mpl
from astropy.io import fits
sys.path.append('../')
import diskdictionary as disk

# target-specific inputs
targets = ['SR4', 'RULup', 'Elias20', 'Sz129', 'HD143006',
           'GWLup', 'Elias24', 'HD163296', 'AS209']

# plotting conventions
rs = 1.2    # plot image extent of +/- (rs * rout) for each target
plt.style.use('classic')
plt.rc('font', size=7)
left, right, bottom, top = 0.06, 0.94, 0.06, 0.98
wspace, hspace = 0.20, 0.25
fig = plt.figure(figsize=(7.5, 5.2))
gs = gridspec.GridSpec(3, 3)

# set colormap
c2 = plt.cm.Reds(np.linspace(0, 1, 32))
c1 = plt.cm.Blues_r(np.linspace(0, 1, 32))
c1 = np.vstack([c1, np.ones((26, 4))])
colors = np.vstack((c1, c2))
mymap = mcolors.LinearSegmentedColormap.from_list('eddymap', colors)
cmap = mymap


# target loop (index i)
for i in range(len(targets)):

    # load residual image
    if np.logical_or(targets[i] == 'HD143006', targets[i] == 'HD163296'):
        hdu = fits.open('data/'+targets[i]+'_resid_symm.JvMcorr.fits')
    else:
        hdu = fits.open('data/'+targets[i]+'_resid.JvMcorr.fits')
    img, hd = np.squeeze(hdu[0].data), hdu[0].header

    # sky-frame cartesian coordinates
    nx, ny = hd['NAXIS1'], hd['NAXIS2']
    RAo  = 3600 * hd['CDELT1'] * (np.arange(nx) - (hd['CRPIX1'] - 1))
    DECo = 3600 * hd['CDELT2'] * (np.arange(ny) - (hd['CRPIX2'] - 1))
    dRA, dDEC = np.meshgrid(RAo - disk.disk[targets[i]]['dx'],
                            DECo - disk.disk[targets[i]]['dy'])

    # disk-frame polar coordinates
    inclr = np.radians(disk.disk[targets[i]]['incl'])
    PAr = np.radians(disk.disk[targets[i]]['PA'])
    xd = (dRA * np.cos(PAr) - dDEC * np.sin(PAr)) / np.cos(inclr)
    yd = (dRA * np.sin(PAr) + dDEC * np.cos(PAr))
    r, theta = np.sqrt(xd**2 + yd**2), np.degrees(np.arctan2(yd, xd))


    # set location in figure
    ax = fig.add_subplot(gs[np.floor_divide(i, 3), i%3])

    # mark the gap(s)
    rgap = disk.disk[targets[i]]['rgap']
    wgap = disk.disk[targets[i]]['wgap']
    for ir in range(len(rgap)):
        ax.fill_between([rgap[ir] - 2 * wgap[ir], rgap[ir] - 2 * wgap[ir],
                         rgap[ir] + 2 * wgap[ir], rgap[ir] + 2 * wgap[ir]],
                        -20, 20, color='darkgray')#, zorder=0)

    # mark the outer edge of the disk
    rout = disk.disk[targets[i]]['rout']
    ax.plot([rout, rout], [-20, 20], '--', color='darkgray', zorder=1)


    # all points!  (on the diverging blue/red colormap)
    resid_snr = 1e6 * img / disk.disk[targets[i]]['RMS']
    sort_indices = np.argsort(resid_snr.flatten())
    s_snr = resid_snr.flatten()[sort_indices]
    s_r = r.flatten()[sort_indices]
    s_snr = s_snr[s_r <= 1.2]
    s_r = s_r[s_r <= 1.2]
    norm = mpl.colors.Normalize(vmin=-10, vmax=10)
    rgba = cmap(norm(s_snr))
  
    
    if (targets[i] == 'SR4'):
        t0 = time.time()
        ax.scatter(s_r, s_snr, c=rgba, marker='o', edgecolors=None, 
                   linewidths=0, rasterized=True) 
        print(time.time() - t0)


    # mean (azimuthally-averaged in fixed radial bins) residual profile
    rbins = np.arange(hd['CDELT2'] * 3600, 1.2, hd['CDELT2'] * 3600)
    dr = np.abs(np.mean(np.diff(rbins)))
    SBr = np.empty(len(rbins))
    for j in range(len(rbins)):
        in_annulus = ((r >= rbins[j] - 0.5 * dr) & (r < (rbins[j] + 0.5 * dr)))
        SBr[j] = np.average(resid_snr[in_annulus])
    ax.plot(rbins, SBr, 'k')



    # limits and labeling
    Rlims = [0, 1.2]
    Tlims = [-10, 10]
    ax.text(0.95, 0.93, disk.disk[targets[i]]['label'], 
            transform=ax.transAxes, ha='right', va='top')
    ax.set_xlim(Rlims)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    ax.set_ylim(Tlims)
    #ax.set_yticks([1, 10, 100])
    #ax.set_yticklabels(['1', '10', '100'])
    if (i == 6):
        ax.set_xlabel('radius  ($^{\prime\prime}$)')
        ax.set_ylabel('residual S/N')
    #if not ((i == 0) or (i == 5)):
    #    ax.set_xticklabels([])
    #    ax.set_yticklabels([])
        

# adjust full figure layout
fig.subplots_adjust(wspace=wspace, hspace=hspace)
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
fig.savefig('../figs/resid_profiles.pdf')
