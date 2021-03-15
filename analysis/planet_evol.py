import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d

plt.rcParams.update({'font.size': 9})


# set up plot
fig = plt.figure(figsize=(3.5, 3.7))
gs = gridspec.GridSpec(2, 1)
axl = fig.add_subplot(gs[0,0])
axr = fig.add_subplot(gs[1,0])

# set up axes, labels
Mlims = [0.005, 20]
Llims = [-9, -2.5]
Rlims = [0, 2.8]
axl.set_xlim(Mlims)
axl.set_xscale('log')
axl.set_ylim(Llims)
axl.set_xticklabels([])
axl.set_ylabel('log ($L_p / L_\odot$)', labelpad=5)
axr.set_xlim(Mlims)
axr.set_xscale('log')
axr.set_xticks([0.01, 0.1, 1, 10])
axr.set_xticklabels(['0.01', '0.1', '1', '10'])
axr.set_xlabel('$M_p$ (M$_{\\rm Jup}$)')
axr.set_ylim(Rlims)
axr.set_ylabel('$R_p$ (R$_{\\rm Jup}$)')


# constants
Ljup = 8.710e-10
Lsun = 3.828e33
Mear = 5.9722e27
Mjup = 1.898e30
Rjup = 6.9911e9
sig_ = 5.67e-5


# load Linder et al. 2019 model grids

# cloudy
mfile = 'Linder19/BEX_evol_mags_-2_MH_0.00_fsed_1.00.dat'
mLAGE, mMPL, mRPL, mLPL = np.loadtxt(mfile, usecols=(0, 1, 2, 3), skiprows=4).T
yo = (mLAGE == 6.0)
M_L19c = mMPL[yo] * Mear / Mjup
L_L19c = np.log10(mLPL[yo] * Ljup)
R_L19c = mRPL[yo]
axl.plot(M_L19c, L_L19c, 'oC1', markersize=4, fillstyle='none')
axr.plot(M_L19c, R_L19c, 'oC1', markersize=4, fillstyle='none')

# clear
mfile = 'Linder19/BEX_evol_mags_-2_MH_0.00.dat'
mLAGE, mMPL, mRPL, mLPL = np.loadtxt(mfile, usecols=(0, 1, 2, 3), skiprows=4).T
wo = (mLAGE == 6.0)
M_L19 = mMPL[wo] * Mear / Mjup
L_L19 = np.log10(mLPL[wo] * Ljup)
R_L19 = mRPL[wo]
axl.plot(M_L19, L_L19, 'oC1', markersize=3)
axr.plot(M_L19, R_L19, 'oC1', markersize=3)



# Spiegel & Burrows 2012 (hot and cold) at 1 Myr
M_SB12 = np.array([1., 2., 5., 10.])
R_SB12hot1  = np.array([1.73, 1.69, 1.85, 2.30])
R_SB12cold1 = np.array([1.41, 1.32, 1.24, 1.14])
T_SB12hot1  = np.array([830., 1200., 1800., 2400.])
T_SB12cold1 = np.array([550., 620., 690., 690.])
L_SB12hot1  = 4*np.pi * sig_ * (R_SB12hot1 * Rjup)**2 * T_SB12hot1**4 / Lsun
L_SB12cold1 = 4*np.pi * sig_ * (R_SB12cold1 * Rjup)**2 * T_SB12cold1**4 / Lsun

axl.plot(M_SB12, np.log10(L_SB12cold1), 'P', color='m', markersize=5)
axl.plot(M_SB12, np.log10(L_SB12hot1), 'X', color='r', markersize=5)
axr.plot(M_SB12, R_SB12cold1, 'P', color='m', markersize=5)
axr.plot(M_SB12, R_SB12hot1, 'X', color='r', markersize=5)


# model means
mean_M = np.concatenate((M_L19[:-1], np.array([1., 2., 5., 10., 20.])))
mean_L, mean_R = np.zeros_like(mean_M), np.zeros_like(mean_M)
mean_L[0] = L_L19[0]
mean_L[1:6] = 0.5*(L_L19[1:6] + L_L19c[0:5])
mean_L[6] = (L_L19[6] + np.log10(L_SB12cold1[0]) + np.log10(L_SB12hot1[0])) / 3.
mean_L[7:10] = 0.5*(np.log10(L_SB12cold1[1:]) + np.log10(L_SB12hot1[1:]))
mean_L[10] = -4
mean_R[0] = R_L19[0]
mean_R[1:6] = 0.5*(R_L19[1:6] + R_L19c[0:5])
mean_R[6] = (R_L19[6] + R_SB12cold1[0] + R_SB12hot1[0]) / 3.
mean_R[7:10] = 0.5*(R_SB12cold1[1:] + R_SB12hot1[1:])
mean_R[10] = 1.9

Lint = interp1d(mean_M, mean_L, kind='quadratic', fill_value='extrapolate')
Rint = interp1d(mean_M, mean_R, kind='quadratic', fill_value='extrapolate')

Mgrid = np.logspace(-2, np.log10(20), 128)
axl.plot(Mgrid, Lint(Mgrid), ':C0')
axr.plot(Mgrid, Rint(Mgrid), ':C0')

np.savez('planetevol.npz', Mgrid=Mgrid, Lgrid=Lint(Mgrid), Rgrid=Rint(Mgrid))


# labeling
axl.plot([0.07], [0.90], 'X', color='r', markersize=5, transform=axl.transAxes)
axl.text(0.10, 0.90, 'SB12 - hot', ha='left', va='center', 
         color='r', transform=axl.transAxes, fontsize=7)
axl.plot([0.07], [0.83], 'P', color='m', markersize=5, transform=axl.transAxes)
axl.text(0.10, 0.83, 'SB12 - cold', ha='left', va='center',
         color='m', transform=axl.transAxes, fontsize=7)
axl.plot([0.07], [0.76], 'oC1', markersize=3, transform=axl.transAxes)
axl.text(0.10, 0.76, 'L19 (solar)', ha='left', va='center',
         color='C1', transform=axl.transAxes, fontsize=7)
axl.plot([0.07], [0.69], 'oC1', markersize=4, fillstyle='none', 
         transform=axl.transAxes)
axl.text(0.10, 0.69, 'L19 (solar, clouds)', ha='left', va='center',
         color='C1', transform=axl.transAxes, fontsize=7)
axl.plot([0.02, 0.09], [0.62, 0.62], ':C0', transform=axl.transAxes)
axl.text(0.10, 0.62, 'adopted', ha='left', va='center', color='C0',
         transform=axl.transAxes, fontsize=7)


fig.subplots_adjust(left=0.13, right=0.87, bottom=0.10, top=0.99, hspace=0.04)
fig.savefig('../figs/planet_evol.pdf')
fig.clf()
