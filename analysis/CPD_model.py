import os, sys
import numpy as np
import scipy.constants as sc
from scipy.interpolate import griddata, interp1d
import matplotlib.pyplot as plt


def CPD_model(Mpl=1.0, Mdot=1e-8, Mcpd=1.0, Tirrs=20., incl=30., kap=2.4, 
              alb=0, p=0.75, dpc=140., rtrunc=0.3, age=1.0, Mstar=1.0, apl=10., 
	      rin=1.0, HR=0.1, Lpl=None, Rpl=None, nu=240, plot_struct=False,
              Rout=None):

    # parse constants
    G_ = sc.G * 1e3
    sig_ = sc.sigma * 1e3
    c_ = sc.c * 1e2
    k_ = sc.k * 1e7
    h_ = sc.h * 1e7

    # parse unit conversions
    yr = sc.year		# s
    au = sc.au * 1e2		# cm
    pc = sc.parsec * 1e2	# cm
    Msun = 1.989e33		# g
    Mjup = 1.898e30		# g
    Mear = 5.9722e27 		# g
    Rjup = 6.9911e9		# cm
    Lsun = 3.828e33		# erg/s (this is the IAU def)
    Ljup = 8.710e-10		# Lsun 

    # get planet properties from specified inputs or models
    # [Lpl] = Lsun, [Rpl] = cm
    if np.logical_and(Lpl is None, Rpl is None):
        pmod = np.load('planetevol.npz')
        Mgrid, Lgrid, Rgrid = pmod['Mgrid'], pmod['Lgrid'], pmod['Rgrid']
        Lint = interp1d(Mgrid, Lgrid, kind='quadratic', 
                        fill_value='extrapolate')
        Rint = interp1d(Mgrid, Rgrid, kind='quadratic',
                        fill_value='extrapolate')
        Lpl, Rpl = 10**(Lint(Mpl)), Rint(Mpl) * Rjup
    else:
        Rpl *= Rjup
    

    # compute CPD radius grid
    Rinn = rin * Rpl 
    if Rout is None:
        Rout = rtrunc * apl*au * (Mpl*Mjup / (3 * Mstar*Msun))**(1./3.)
    else:
        Rout *= au
    r = np.logspace(np.log10(Rinn), np.log10(Rout), 256)
    r[0] = Rinn

    # compute viscous heating profile
    Tacc4 = 3 * G_ * Mpl*Mjup * Mdot*(Mjup/yr) * (1. - (Rinn / r)**0.5) / \
            (8 * np.pi * sig_ * r**3)

    # compute planet irradiation heating profile
    Tpl4 = HR * Lpl*Lsun / (4 * np.pi * sig_ * r**2)

    # compute CPD temperature profile
    Tcpd = (Tacc4 + Tpl4 + Tirrs**4)**0.25

    # compute CPD surface density profile
    Sigma_out = (2 - p) * Mcpd*Mear / \
                (2 * np.pi * Rout**p * (Rout**(2-p) - Rinn**(2-p)))
    Sigma = Sigma_out * (Rout / r)**p

    # if desired, plot the CPD structure
    if plot_struct:
        fig, axs = plt.subplots(nrows=2, figsize=(4, 5))

        # temperature profiles
        ax = axs[0]
        ax.plot(r/au, Tacc4**0.25, '--C0')
        ax.plot(r/au, Tpl4**0.25, '--C1')
        ax.plot(r/au, np.ones_like(r) * Tirrs, '--C2')
        ax.plot(r/au, Tcpd, 'k', lw=2)
        ax.set_xlim([Rinn/au, Rout/au])
        ax.set_ylim([2, 2 * Tcpd.max()])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('temperature  (K)')
        ax.text(0.93, 0.90, 'Tcpd', color='k', ha='right', va='center',
                transform=ax.transAxes)
        ax.text(0.93, 0.82, 'Tacc', color='C0', ha='right', va='center',
                transform=ax.transAxes)
        ax.text(0.93, 0.74, 'Tirr,pl', color='C1', ha='right', va='center',
                transform=ax.transAxes)
        ax.text(0.93, 0.66, 'Tirr,*', color='C2', ha='right', va='center',
                transform=ax.transAxes)

        # surface density profile
        ax = axs[1]
        ax.plot(r/au, Sigma, 'k', lw=2)
        ax.set_xlim([Rinn/au, Rout/au])
        ax.set_ylim([0.5 * Sigma.min(), 2 * Sigma.max()])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('radius  (au)')
        ax.set_ylabel('surface density  (g/cm$^2$)')


        fig.subplots_adjust(left=0.15, right=0.85, bottom=0.10, top=0.98)
        fig.subplots_adjust(hspace=0.20)
        fig.savefig('CPD_model.structure.png')
        fig.clf()


    # compute the optical depths
    tau = kap * Sigma / (1. - alb)

    # compute the scattering correction term
    mu  = np.cos(np.radians(incl))
    eps = np.sqrt(1. - alb)
    num1 = (1 - np.exp(-(np.sqrt(3) * eps + 1 / mu) * tau)) / \
           (np.sqrt(3) * eps * mu + 1)
    num2 = (np.exp(-tau / mu) - np.exp(-np.sqrt(3) * eps * tau)) / \
           (np.sqrt(3) * eps * mu - 1)
    den  = np.exp(-np.sqrt(3) * eps * tau) * (eps - 1) - (eps + 1)
    FF = (num1 + num2) / den

    # compute the flux density in uJy
    Bnu = (2 * h_ * (nu*1e9)**3 / c_**2) / \
          (np.exp(h_ * nu*1e9 / (k_ * Tcpd)) - 1.)
    SB = Bnu * ((1 - np.exp(-tau / mu)) + alb * FF)
    Fcpd = 1e29 * (2 * np.pi * mu / (dpc*pc)**2) * np.trapz(SB * r, r)

    return Fcpd
