import os, sys
import numpy as np

# input parameters
R_in = 1.0	# CPD inner edge in Rjup
R_out = 1.85	# CPD outer edge in au
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
R_ = 8.3145e7		# gas constant in erg / K / mol
au_ = 1.496e13		# au in cm
Msun = 1.989e33		# solar mass in g
Mjup = 1.898e30		# Jupiter mass in g
Rjup = 6.9911e9		# Jupiter radius in cm
yr = 3.154e7		# year in seconds


# radius grid
rau = np.logspace(np.log10(R_in * Rjup / au), np.log10(R_out), 1024)
r = rau * au



