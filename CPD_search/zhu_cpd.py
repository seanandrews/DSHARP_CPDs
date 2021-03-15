import os, sys
import numpy as np
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt

# input parameters
R_in = 1.0	# CPD inner edge in Rjup
R_out = 5.  	# CPD outer edge in au
R_p = 1.0	# planet radius in Rjup
M_p = 1.0	# planet mass in Mjup
Mdot = 1.0e-5	# planet accretion rate in Mjup / yr
alpha = .001	# turbulence coefficient
wl = 1.25	# wavelength in mm
dpc = 140.	# distance in pc



# constants
G_ = 6.67e-8		# gravitational constant in cm**2 / g / s**2
k_ = 1.3801e-16		# Boltzmann constant in erg / K
sig_ = 5.67e-5		# Stefan-Boltzmann constant in erg / cm**2 / s / K**4
mu = 2.4		# mean molecular weight of CPD material
mH = 1.67e-27           # H mass in g
au_ = 1.496e13		# au in cm
Msun = 1.989e33		# solar mass in g
Mjup = 1.898e30		# Jupiter mass in g
Rjup = 6.9911e9		# Jupiter radius in cm
yr = 3.154e7		# year in seconds


# radius grid
rau = np.logspace(np.log10(R_in * Rjup / au_), np.log10(R_out), 1024)
r = rau * au_

# opacities
kappa_mm = 0.034 * (0.87 / wl)
kappa_R = 10.

# effective temperature (Equation 1)
Teff4 = 3 * G_ * (M_p * Mjup) * (Mdot * Mjup / yr) * (1 - r[0] / r)**0.5 / \
        (8 * np.pi * sig_ * r**3)

# irradiation temperature (Equation 3)
Lirr = G_ * (M_p * Mjup) * (Mdot * Mjup / yr) / (2 * R_p * Rjup)
Tirr4 = Lirr / (40 * np.pi * sig_ * r**2)

# external temperature
Tism = 10.
Text4 = Tirr4 + Tism**4

# midplane temperature
Tc4 = 

# surface density (Equation 6) for viscous heating-dominated models
sigma_6 = (2**1.4 / 3**1.2) * \
          (sig_ * G_ * (M_p * Mjup) * (Mdot * Mjup / yr)**3 / \
           (alpha**4 * np.pi**3 * kappa_R * r**3))**0.2 * (mu / R_)**0.8

# surface density (Equation 7) for irradiating heating-dominated models
om = np.sqrt(G_ * M_p * Mjup / r**3)
sigma_7 = (Mdot * Mjup / yr) * mu * om / (3 * np.pi * alpha * R_ * Text4**0.25)


sigma = sigma_6


# midplane temperature
Tc4 = 9 * G_ * (M_p * Mjup) * (Mdot * Mjup / yr) * sigma * kappa_R * \
      (1 - r[0] / r)**0.5 / (128 * np.pi * sig_ * r**3) + Text4

# mm optical depths
tau_mm = kappa_mm * sigma / 2

# brightness temperatures (Equation 8)
Tb = ((3 * kappa_R / (8 * kappa_mm)) * Teff4 + Text4)**0.25
Tb[tau_mm < 0.5] = 2 * Tc4[tau_mm < 0.5]**0.25 * tau_mm[tau_mm < 0.5]


# surface brightness profile (Jy per arcsec**2)
Inu = 1e23 * 2 * k_ * Tb / (wl * 0.1)**2 / (180 * 3600 / np.pi)**2

# integrate to get flux density as a function of outer radius
flux_r = cumtrapz(2 * np.pi * (rau / dpc) * Inu, (rau / dpc), initial=1e-50)



fig, ax = plt.subplots()
ax.plot(np.log10(rau), np.log10(sigma), 'k')
ax.grid()
ax.set_xlim([-3.5, 0.5])
ax.set_ylim([1.5, 4.5])
ax.set_xlabel('log(R / AU)')
ax.set_ylabel('log(Sigma / g/cm**2)')
plt.show()

fig, ax = plt.subplots()
ax.plot(np.log10(rau), np.log10(Tc4**0.25), 'r')
ax.plot(np.log10(rau), np.log10(Teff4**0.25), 'b')
ax.plot(np.log10(rau), np.log10(Tb), '--k')
ax.grid()
ax.set_xlim([-3.5, 0.5])
ax.set_ylim([1.0, 4.5])
ax.set_xlabel('log(R / AU)')
ax.set_ylabel('log(T / K)')
plt.show()





