import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from astropy.io import fits
sys.path.append('../')
import diskdictionary as disk

# target-specific inputs
targets = ['SR4', 'RULup', 'Elias20', 'Sz129', 'HD143006',
           'GWLup', 'Elias24', 'HD163296', 'AS209']
glbls = ['D11', 'D29', 'D25', 'D41', 'D64', 'D22', 'D51', 'D74', 'D57', 'D48', 'D86', 'D61', 'D97']

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
lix = 0
for i in range(len(targets)):

    # set the location in the figure
    ax = fig.add_subplot(gs[np.floor_divide(i, 3), i%3])

    # how many gaps does this target have?
    ngaps = len(disk.disk[targets[i]]['rgap'])

    # loop through the recovery profiles for each gap
    for ig in range(ngaps):
        spl_flag = False
        
        # check and see if there is a recovery profile; if not, make a dummy
        rfile = 'recoveries/'+targets[i]+'_gap'+str(ig)+'_rprofs'
        if os.path.exists(rfile+'.all.txt'):
            infile = rfile+'.all.txt'
            Fi, recA, erecA, falseA, recB, erecB, falseB = np.loadtxt(infile).T
            spl_flag = True
        elif os.path.exists(rfile+'.0.txt'):
            infile = rfile+'.0.txt'
            Fi, recA, erecA, falseA, recB, erecB, falseB = np.loadtxt(infile).T
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
            col = '#1f77b4'
            lpos = 0.95
        else:
            col = '#d62728'
            lpos = 0.83
        ax.errorbar(Fi, rec, yerr=erec, marker='.', color=col, ls='none',
                    markersize=6, capsize=0.0, elinewidth=1.5)

        # interpolation for nice curves to guide the eye
        if spl_flag:
            rint = interp1d(np.append(np.append(rec, 0), 1.0), 
                            np.append(np.append(Fi, 0), 260), kind='linear') 
            xx = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            print(' ')
            print(targets[i] + '_gap' + str(ig) + ' fluxes for recovery %iles')
            print(xx)
            print(['%i' % j for j in rint(xx)])

        # gap labeling
        #if (glbls[lix] == 'D61'): lpos = 0.79
        ax.text(lpos, 0.06, glbls[lix], color=col, transform=ax.transAxes, 
                ha='right', va='bottom')
        lix += 1
            

    # limits and labeling
    ax.text(0.95, 0.15, disk.disk[targets[i]]['label'], 
            transform=ax.transAxes, ha='right', va='bottom')
    ax.set_xlim(Flims)
    ax.set_ylim(Rlims)
    if (i == 6):
        ax.set_xlabel('$F_{\\rm cpd}$  ($\\mu {\\rm Jy}$)')
        ax.set_ylabel('recovery fraction')
        

# adjust full figure layout
fig.subplots_adjust(wspace=wspace, hspace=hspace)
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
fig.savefig('../figs/recov_profiles.pdf')
