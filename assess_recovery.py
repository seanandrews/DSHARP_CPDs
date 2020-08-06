import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# load the entries in the recoveries file
rfile = 'SR4_continuum_CPDrecoveries.txt'
Fi, Fo, mdl, ri, ro, azi, azo, SNR = np.loadtxt(rfile).T


# recovery criteria
is_rec = ((SNR >= 3.) & (((Fo - Fi) / (Fo / SNR)) < 3.) & \
          (((ro - ri) / 0.035) < 0.2) & (((azo - azi) / 25.) < 0.2))
nFs = len(np.unique(Fi))
for i in range(nFs):
    recovd = Fi[is_rec & (Fi == Fi[i])]
    print(len(recovd))



# SNR as a function of Fi
fig, ax = plt.subplots()
ax.plot(Fi, SNR, 'o', markersize=2)
ax.plot([90, 300], [3, 3], '--k', alpha=0.5)
ax.set_xlim([90, 300])
ax.set_ylim([0, 12])
ax.set_xlabel('input CPD flux [$\mu$Jy]')
ax.set_ylabel('recovery SNR')
fig.savefig('recoveries/Fi_SNR.png')

# fractional recovered F as a function of Fi
fig, ax = plt.subplots()
ax.plot(Fi, 1 + (Fo - Fi) / Fi, 'o', markersize=2)
ax.plot([90, 300], [1, 1], '--k', alpha=0.5)
ax.set_xlim([90, 300])
ax.set_ylim([0, 2])
ax.set_xlabel('input CPD flux [$\mu$Jy]')
ax.set_ylabel('fractional recovered flux')
fig.savefig('recoveries/Fi_fracFrec.png')

# deviation of recovered F in noise units, as a function of Fi
fig, ax = plt.subplots()
ax.plot(Fi, (Fo - Fi) / (Fo / SNR), 'o', markersize=2)
ax.plot([90, 300], [0, 0], '--k', alpha=0.5)
ax.set_xlim([90, 300])
ax.set_ylim([-5, 5])
ax.set_xlabel('input CPD flux [$\mu$Jy]')
ax.set_ylabel('recovered flux deviation in noise units')
fig.savefig('recoveries/Fi_recFnoise.png')

# Fi versus Fo (simple)
fig, ax = plt.subplots()
ax.plot(Fi, Fo, 'o', markersize=2)
ax.plot([0, 400], [0, 400], '--k', alpha=0.5)
ax.set_xlim([50, 350])
ax.set_ylim([50, 350])
ax.set_xlabel('input CPD flux [$\mu$Jy]')
ax.set_ylabel('recovered CPD flux [$\mu$Jy]')
fig.savefig('recoveries/Fi_Fo.png')

# azi versus azo (simple)
norm = mpl.colors.Normalize(vmin=100, vmax=250)
cmap = mpl.cm.get_cmap('viridis')
fig, ax = plt.subplots()
for i in range(len(np.unique(Fi))):
    ax.plot(azi[Fi == np.unique(Fi)[i]], azo[Fi == np.unique(Fi)[i]], 'o', 
            color=cmap(norm(np.unique(Fi)[i])), markersize=2)
ax.plot([-185, 185], [-185, 185], '--k', alpha=0.5)
ax.set_xlim([-185, 185])
ax.set_ylim([-185, 185])
ax.set_xlabel('input CPD azimuth [degr]')
ax.set_ylabel('recovered CPD azimuth [degr]')
fig.savefig('recoveries/azi_azo.png')

# fractional recovered az as a function of azi
fig, ax = plt.subplots()
ax.plot(Fi, (azo - azi) / 25., 'o', markersize=2)
ax.plot([90, 300], [0., 0.], '--k', alpha=0.5)
ax.set_xlim([90, 300])
ax.set_ylim([-3, 3])
ax.set_xlabel('input CPD flux [$\mu$Jy]')
ax.set_ylabel('$\Delta$ recovered azimuth (fraction of beam)')
fig.savefig('recoveries/azi_fracazrec.png')



