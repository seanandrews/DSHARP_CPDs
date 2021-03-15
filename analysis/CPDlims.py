import os, sys, time
import numpy as np
from derive_CPDlims import derive_CPDlims


# set up Mpl and Rth grids
Mpl = np.logspace(np.log10(0.02), np.log10(20.), 80)
Rth = np.linspace(0.1, 0.8, 68)

# target-specific lists
targets = ['SR4', 'RULup', 'Elias20', 'Sz129', 'Sz129', 'HD143006', 'HD143006',
           'GWLup', 'Elias24', 'HD163296', 'HD163296', 'AS209', 'AS209']
gap_ixs = [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1]
Flims = [118., 55., 60., 44., 53., 71., 85., 55., 62., 124., 112., 56., 70.]


# model-specific iterations
kabs = [2.4, 2.4]
alb = [0.0, 0.9]
glbls = ['noscat', 'wscat']

# target loop
for i in range(len(targets)):

    for j in range(len(glbls)):

        t0 = time.time()
        print(targets[i]+'_gap'+str(gap_ixs[i]) + ' '+glbls[j] + ' ...')

        Mlims = derive_CPDlims(targets[i], gap_ixs[i], Flims[i], Mpl, Rth, 
                               eps_Mdot=1, kap=kabs[j], alb=alb[j])

        ofile = targets[i]+'_gap'+str(gap_ixs[i])+'_'+glbls[j]+'.Mlims'
        np.savez('Mlims_data/'+ofile+'.npz', Mpl=Mpl, Rth=Rth, Mlims=Mlims)
        print(time.time()-t0)
