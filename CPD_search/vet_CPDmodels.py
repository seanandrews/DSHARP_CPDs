import os, sys
import numpy as np
import matplotlib.pyplot as plt
from CPD_zhu18 import CPD_zhu18

# load Zhu et al. 2018 Table 1
z_MpMdot, z_alpha, z_Rout, z_mass = np.loadtxt('zhu_table1.txt', 
                                               usecols=(0,1,2,3)).T

z_F7nob, z_F7b = np.loadtxt('zhu_table1.txt', usecols=(5,6)).T

# fixed CPD parameters
Rin = 1.0	# inner radius in Rjup units
Mp = 1.0
Rp = 1.0	# planet radius in Rjup units
wl = 0.85	# wavelength in mm
dpc = 140.	# distance in pc

# compute CPD models
my_mass = np.zeros_like(z_mass)
my_F7 = np.zeros_like(z_mass)
for i in range(len(z_MpMdot)):

    # compute CPD model
    CPD = CPD_zhu18(Mp=Mp, Mdot=z_MpMdot[i] / Mp, Rin=Rin, Rout=z_Rout[i], 
                    Rp=Rp, alpha=z_alpha[i], wl=wl, dpc=dpc, method='viscous')

    # get the masses
    my_mass[i] = CPD.mass

    # get the B7 flux
    my_F7[i] = CPD.flux


fig, axs = plt.subplots(ncols=2, figsize=(7.5, 3.5))

ax = axs[0]
ax.plot(z_mass, my_mass, 'o')
mdlx = np.logspace(-7, 1, 10)
ax.plot(mdlx, mdlx, '--k')
ax.plot(mdlx, 1.25*mdlx, ':r')
ax.set_xlim([2e-6, 5.])
ax.set_ylim([2e-6, 5.])
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('CPD mass from Zhu+ 2018 Table 1')
ax.set_ylabel('CPD mass from my calculation')

ax = axs[1]
ax.plot(z_F7b, my_F7, 'o')
mdlx = np.logspace(-2, 4, 10)
ax.plot(mdlx, mdlx, '--k')
ax.set_xlim([0.02, 5000])
ax.set_ylim([0.02, 5000])
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('CPD flux from Zhu+ 2018 Table 1')
ax.set_ylabel('CPD flux from my calculation')

fig.subplots_adjust(wspace=0.3, left=0.08, right=0.92, bottom=0.13, top=0.99)
fig.savefig('mass_FB7_table1.png')



