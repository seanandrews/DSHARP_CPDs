import os, sys
import numpy as np
from scipy.integrate import cumtrapz
import scipy.constants as sc
from scipy.interpolate import interp1d

# input parameters
Rin = 1.0	# CPD inner edge in Rjup
Rout = 5.  	# CPD outer edge in au
Rp = 1.0	# planet radius in Rjup
Mp = 1.0	# planet mass in Mjup
Mdot = 1.0e-5	# planet accretion rate in Mjup / yr
alpha = .001	# turbulence coefficient
wl = 1.25	# wavelength in mm
dpc = 140.	# distance in pc


def CPD_zhu18(Mp=1.0, Mdot=1e-5, Rin=1.0, Rout=1.0, Rp=1.0, alpha=0.001, 
              wl=1.25, dpc=140., method='viscous'):

    # constants
    G = sc.G * 1e3	   # gravitational constant in cm**2 / g / s**2
    k = sc.k * 1e7	   # Boltzmann constant in erg / K
    sig = sc.sigma * 1e3   # Stefan-Boltzmann constant in erg / cm**2 / s / K**4
    mu = 2.4	     	   # mean molecular weight of CPD material
    RR = sc.R * 1e7	   # gas constant in erg / K / mol
    au = sc.au * 1e2	   # au in cm
    Msun = 1.989e33	   # solar mass in g
    Mjup = 1.898e30	   # Jupiter mass in g
    Rjup = 6.9911e9	   # Jupiter radius in cm
    yr = sc.year	   # year in seconds

    # convert relevant parameters into cgs units
    Ri_ = Rin * Rjup
    Ro_ = Rout * au
    Rp_ = Rp * Rjup
    Mp_ = Mp * Mjup
    Mdot_ = Mdot * Mjup / yr

    # radius grid
    r = np.logspace(np.log10(Ri_), np.log10(10 * au), 4096)
    rau = r / au

    # opacities
    kappa_mm = 0.034 * (0.87 / wl)
    kappa_R = 10.

    # effective temperature (Equation 1)
    Teff4 = 3 * G * Mp_ * Mdot_ * (1 - r[0] / r)**0.5 / (8*np.pi * sig * r**3)

    # irradiation temperature (Equation 3)
    Lirr = G * Mp_ * Mdot_ / (2 * Rp_)
    Tirr4 = Lirr / (40 * np.pi * sig * r**2)

    # external temperature
    Tism = 10.
    Text4 = Tirr4 + Tism**4

    # surface density (Equation 6) for viscous heating-dominated models
    sigma_6 = (2**1.4 / 3**1.2) * (sig * G * Mp_ * Mdot_**3 / \
              (alpha**4 * np.pi**3 * kappa_R * r**3))**0.2 * (mu / RR)**0.8 * \
              (1 - (r[0] / r)**0.5)**0.6

    # surface density (Equation 7) for irradiating heating-dominated models
    om = np.sqrt(G * Mp_ / r**3)
    sigma_7 = Mdot_ * mu * om / (3 * np.pi * alpha * RR * Text4**0.25)

    # composite 
    if (method == 'viscous'):
        sigma = sigma_6
    if (method == 'irradiated'):
        sigma = sigma_7
    if (method == 'composite'):
        sigma = np.minimum(sigma_6, sigma_7)

    # integrate to get mass as a function of radius
    mass_r = cumtrapz(2*np.pi * r * sigma, r, initial=0)

    # interpolate to get mass at a fixed outer radius
    mint = interp1d(r, mass_r)
    mass_cpd = mint(Ro_)

    # midplane temperature
    Tc4 = 9 * G * Mp_ * Mdot_ * sigma * kappa_R * (1 - r[0] / r)**0.5 / \
          (128 * np.pi * sig * r**3) + Text4

    # mm optical depths
    tau_mm = kappa_mm * sigma / 2

    # brightness temperatures (Equation 8)
    Tb = ((3 * kappa_R / (8 * kappa_mm)) * Teff4 + Text4)**0.25
    Tb[tau_mm < 0.5] = 2 * Tc4[tau_mm < 0.5]**0.25 * tau_mm[tau_mm < 0.5]

    # surface brightness profile (Jy per arcsec**2)
    Inu = 1e23 * 2 * k * Tb / (wl * 0.1)**2 / (180 * 3600 / np.pi)**2

    # integrate to get flux density as a function of radius
    flux_r = cumtrapz(2*np.pi * (rau / dpc) * Inu, (rau / dpc), initial=1e-50)

    # interpolate to get flux density at a fixed outer radius
    fint = interp1d(r, flux_r)
    flux_cpd = fint(Ro_)

    # prepare outputs
    class CPD_model:
        def __init__(self, params, flux, r, Tirr, Teff, Tc, sig6, sig7, sigma,
                     mass, method):
            self.Mp = params[0]
            self.Mdot = params[1]
            self.Rin = params[2]
            self.Rout = params[3]
            self.Rp = params[4]
            self.alpha = params[5]
            self.wl = params[6]
            self.distance = params[7]
            self.flux = flux
            self.r = r
            self.Tirr = Tirr
            self.Teff = Teff
            self.Tc = Tc
            self.sig6 = sig6
            self.sig7 = sig7
            self.sigma = sigma
            self.mass = mass
            self.method = method

    output = CPD_model([Mp, Mdot, Rin, Rout, Rp, alpha, wl, dpc], 
                       1e6*flux_cpd, r, Tirr4**0.25, Teff4**0.25, Tc4**0.25, 
                       sigma_6, sigma_7, sigma, mass_cpd / Mjup, method)

    return output
