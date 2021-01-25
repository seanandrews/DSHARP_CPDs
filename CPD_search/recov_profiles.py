import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import fits
sys.path.append('../')
import diskdictionary as disk

# target-specific inputs
targets = ['SR4', 'RULup', 'Elias20', 'Sz129', 'HD143006',
           'GWLup', 'Elias24', 'HD163296', 'AS209']

Rlims  = [0.0, 1.0]
Flims  = [0, 255]
method = 'B'

# plotting conventions
plt.style.use('classic')
plt.rc('font', size=7)
left, right, bottom, top = 0.06, 0.94, 0.06, 0.98
wspace, hspace = 0.20, 0.25
fig = plt.figure(figsize=(7.5, 5.2))
gs = gridspec.GridSpec(3, 3)


# target loop (index i)
for i in range(len(targets)):

    # set the location in the figure
    ax = fig.add_subplot(gs[np.floor_divide(i, 3), i%3])

    # how many gaps does this target have?
    ngaps = len(disk.disk[targets[i]]['rgap'])

    # loop through the recovery profiles for each gap
    for ig in range(ngaps):
        
        # check and see if there is a recovery profile; if not, make a dummy
        rfile = 'recoveries/'+targets[i]+'_gap'+str(ig)+'_rprofs.0.txt'
        if os.path.exists(rfile):
            Fi, recA, erecA, falseA, recB, erecB, falseB = np.loadtxt(rfile).T
        else:
            Fi = np.arange(10, 260, 10)
            recA, erecA = np.zeros_like(Fi), np.zeros_like(Fi)
            recB, erecB = np.zeros_like(Fi), np.zeros_like(Fi)

        # choose the method for recovery (A or B)
        if (method == 'A'):
            rec, erec = recA, erecA
        else:
            rec, erec = recB, erecB

        # plot the recovery profile for each gap
        # (like in resid_maps; inner gap is gray, outer is black)
        if np.logical_or((ngaps == 1), ig == 1):
            col = 'k'
        else:
            col = 'dimgrey'
        ax.errorbar(Fi, rec, yerr=erec, marker='.', color=col, ls='none',
                    markersize=6, capsize=0.0, elinewidth=1.5)

    # limits and labeling
    ax.text(0.05, 0.92, disk.disk[targets[i]]['label'], 
            transform=ax.transAxes, ha='left', va='top')
    ax.set_xlim(Flims)
    ax.set_ylim(Rlims)
    if (i == 6):
        ax.set_xlabel('$F_{\\rm cpd}$  ($\\mu {\\rm Jy}$)')
        ax.set_ylabel('recovery fraction')
        

# adjust full figure layout
fig.subplots_adjust(wspace=wspace, hspace=hspace)
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
fig.savefig('../figs/recov_profiles.pdf')
