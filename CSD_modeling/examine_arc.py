import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from astropy.io import fits
from astropy.visualization import (AsinhStretch, LinearStretch, ImageNormalize)
sys.path.append('../')
import diskdictionary as disk


# controls
target = 'HD143006'
rbounds = [0.38, 0.55]
azbounds = [90., 142.]

#target = 'HD163296'
#rbounds = [0.48, 0.60]
#azbounds = [50., 150.]

# constants
c_ = 2.99792e10
k_ = 1.38057e-16

# residuals diverging color map
c2 = plt.cm.Reds(np.linspace(0, 1, 32))
c1 = plt.cm.Blues_r(np.linspace(0, 1, 32))
c1 = np.vstack([c1, np.ones((26, 4))])
colors = np.vstack((c1, c2))
mymap = mcolors.LinearSegmentedColormap.from_list('eddymap', colors)

plt.style.use('default')
plt.rc('font', size=21)
xyticks = [-0.5, 0, 0.5]


# files
im_files = ['_data.cleanmodel', '_data_arc.cleanmodel',
            '_data.JvMcorr', '_data_symm.JvMcorr',
            '_model.JvMcorr', '_model_symm.JvMcorr',
            '_resid.JvMcorr', '_resid_symm.JvMcorr',
            '_resid.JvMcorr', '_resid_symm.JvMcorr']

for i in range(len(im_files)):

    # load data
    hdu = fits.open('data/'+target+im_files[i]+'.fits')
    img, hd = np.squeeze(hdu[0].data), hdu[0].header

    # parse coordinate frame indices into physical numbers
    RA = 3600 * hd['CDELT1'] * (np.arange(hd['NAXIS1']) - (hd['CRPIX1'] - 1)) 
    DEC = 3600 * hd['CDELT2'] * (np.arange(hd['NAXIS2']) - (hd['CRPIX2'] - 1))
    dRA, dDEC = np.meshgrid(RA - disk.disk[target]['dx'], 
                            DEC - disk.disk[target]['dy'])
    freq = hd['CRVAL3']

    # disk-frame polar coordinates
    inclr = np.radians(disk.disk[target]['incl'])
    PAr = np.radians(disk.disk[target]['PA'])
    xd = (dRA * np.cos(PAr) - dDEC * np.sin(PAr)) / np.cos(inclr)
    yd = (dRA * np.sin(PAr) + dDEC * np.cos(PAr))
    r, theta = np.sqrt(xd**2 + yd**2), np.degrees(np.arctan2(yd, xd))

    # beam parameters
    if (i >= 2):
        bmaj, bmin, bPA = 3600 * hd['BMAJ'], 3600 * hd['BMIN'], hd['BPA']
        res_area = (np.pi*bmaj*bmin / (4*np.log(2))) / (3600*180/np.pi)**2
    else:
        res_area = np.abs(hd['CDELT1'] * hd['CDELT2']) / (180 / np.pi)**2

    # image setups
    rout = disk.disk[target]['rout']
    print(rout)
    im_bounds = (dRA.max(), dRA.min(), dDEC.min(), dDEC.max())
    dRA_lims, dDEC_lims = [1.15*rout, -1.15*rout], [-1.15*rout, 1.15*rout]
    print(dRA_lims, dDEC_lims)

    # intensity limits, and stretch
    if (i >= 8):
        norm = ImageNormalize(vmin=-10, vmax=10, stretch=LinearStretch())
        cmap = mymap
        SB = 1e6 * img / disk.disk[target]['RMS']
    else:
        norm = ImageNormalize(vmin=0, vmax=disk.disk[target]['maxTb'], 
                              stretch=AsinhStretch())
        cmap = 'inferno'
        SB = (1e-23 * img / res_area) * c_**2 / (2 * k_ * freq**2)

    ### Plot the data image
    fig = plt.figure(figsize=(7.75, 6.2))
    gs  = gridspec.GridSpec(1, 2, width_ratios=(1, 0.04))
    ax = fig.add_subplot(gs[0,0])
    im = ax.imshow(SB, origin='lower', cmap=cmap, extent=im_bounds, 
                   norm=norm, aspect='equal')

    # annotations
    tbins = np.linspace(-np.pi, np.pi, 181)
    
    # bounds of asymmetry
    tazbins = np.linspace(np.radians(azbounds[0]), 
                          np.radians(azbounds[1]), 101)
    xai = rbounds[0] * np.cos(tazbins) * np.cos(inclr) 
    yai = rbounds[0] * np.sin(tazbins)
    xao = rbounds[1] * np.cos(tazbins) * np.cos(inclr)
    yao = rbounds[1] * np.sin(tazbins)
    xarci =  xai * np.cos(PAr) + yai * np.sin(PAr)
    yarci = -xai * np.sin(PAr) + yai * np.cos(PAr)
    xarco =  xao * np.cos(PAr) + yao * np.sin(PAr)
    yarco = -xao * np.sin(PAr) + yao * np.cos(PAr)
    if (i >= 8):
        asym_col = 'k'
    else:
        asym_col = 'w'
    ax.plot(xarci, yarci, asym_col, lw=3)
    ax.plot(xarco, yarco, asym_col, lw=3)
    ax.plot([xarci[0], xarco[0]], [yarci[0], yarco[0]], asym_col, lw=3)
    ax.plot([xarci[-1], xarco[-1]], [yarci[-1], yarco[-1]], asym_col, lw=3)

    #if (i >= 8):
    #    rgapi = disk.disk[target]['rgapi'][::-1]
    #    rgapo = disk.disk[target]['rgapo'][::-1]
    #    gcols = ['k', 'darkgray']
    #    for ir in range(len(rgapi)):
    #        xgi = rgapi[ir] * np.cos(tbins) * np.cos(inclr)
    #        ygi = rgapi[ir] * np.sin(tbins)
    #        ax.plot( xgi * np.cos(PAr) + ygi * np.sin(PAr),
    #                -xgi * np.sin(PAr) + ygi * np.cos(PAr), '-', 
    #                color=gcols[ir])
    #        xgo = rgapo[ir] * np.cos(tbins) * np.cos(inclr)
    #        ygo = rgapo[ir] * np.sin(tbins)
    #        ax.plot( xgo * np.cos(PAr) + ygo * np.sin(PAr),
    #                -xgo * np.sin(PAr) + ygo * np.cos(PAr), '-', 
    #                color=gcols[ir])
#
#        xout, yout = rout * np.cos(tbins) * np.cos(inclr), rout * np.sin(tbins)
#        ax.plot( xout * np.cos(PAr) + yout * np.sin(PAr),
#                -xout * np.sin(PAr) + yout * np.cos(PAr), ':', color='gray')

#    if (i < 2):
#        ri, ro = rbounds[0], rbounds[1]
#        xi, yi = ri * np.cos(tbins) * np.cos(inclr), ri * np.sin(tbins)
#        xo, yo = ro * np.cos(tbins) * np.cos(inclr), ro * np.sin(tbins)
#        ax.plot( xi * np.cos(PAr) + yi * np.sin(PAr),
#                -xi * np.sin(PAr) + yi * np.cos(PAr), ':w')
#        ax.plot( xo * np.cos(PAr) + yo * np.sin(PAr),
#                -xo * np.sin(PAr) + yo * np.cos(PAr), ':w')
        
    # beam
    if (i >= 2):
        beam = Ellipse((dRA_lims[0] + 0.1*np.diff(dRA_lims), 
                        dDEC_lims[0] + 0.1*np.diff(dDEC_lims)), 
                        bmaj, bmin, 90-bPA)
        if (i < 8):
            beam.set_facecolor('w')
        else:
            beam.set_facecolor('k')
        ax.add_artist(beam)

    # limits, labeling
    ax.set_xlim(dRA_lims)
    ax.set_xticks(xyticks[::-1])
    ax.set_ylim(dDEC_lims)
    ax.set_yticks(xyticks)
    ax.set_xlabel('RA offset  ($^{\prime\prime}$)')
    ax.set_ylabel('DEC offset  ($^{\prime\prime}$)', labelpad=-6)

    # add a scalebar
    cbax = fig.add_subplot(gs[:,1])
    if (i < 8):
        cb = Colorbar(ax=cbax, mappable=im, orientation='vertical',
                      ticklocation='right')
        cb.set_label('brightness temperature  (K)', rotation=270, labelpad=21)
    else:
        cb = Colorbar(ax=cbax, mappable=im, orientation='vertical',
                      ticklocation='right', 
                      ticks=[-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
        cb.set_label('residual S/N', rotation=270, labelpad=21)
                     

    # adjust layout
    fig.subplots_adjust(wspace=0.00)
    fig.subplots_adjust(left=0.115, right=0.87, bottom=0.11, top=0.98)
    ofile = '../figs/'+target+'_demo'+im_files[i]
    if (i >= 8):
        ofile += '_div'
    fig.savefig(ofile+'.pdf')
    fig.clf()
