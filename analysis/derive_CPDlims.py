import os, sys, time
import numpy as np
import scipy.constants as sc
from scipy.interpolate import interp1d
from astropy.io import fits
from CPD_model import CPD_model
sys.path.append('../')
import diskdictionary as disk


def derive_CPDlims(target, gap_ix, Flim, Mpl, Rth, eps_Mdot=1, 
                   kap=2.4, alb=0.0):

    # always-fixed parameters
    incl  = disk.disk[target]['incl']		# CPD incl = disk incl
    psl   = 0.75				# CPD surface density index
    dpc   = disk.disk[target]['distance']	# distance in pc
    Mstar = disk.disk[target]['mstar']		# stellar mass
    Lstar = disk.disk[target]['lstar']		# stellar luminosity
    rin   = 1.0
    HR    = 0.1
    dtog  = 0.01

    # constants
    sig_ = sc.sigma * 1e3
    au = sc.au * 1e2
    Lsun = 3.828e33

    # additional fixed parameters that need to be computed / retrieved
    # planet location
    apl = disk.disk[target]['rgap'][gap_ix] * dpc

    # continuum frequency
    imfile = '../CSD_modeling/data/'+target+'_data.JvMcorr.fits'
    hd = fits.open(imfile)[0].header
    nu = hd['CRVAL3'] / 1e9

    # stellar/disk irradiation heating term
    if (target == 'RULup'):
        phi = 0.05
    else:
        phi = 0.02
    Tirrs = (phi * Lstar*Lsun / (8 * np.pi * sig_ * (apl*au)**2))**0.25


    # iterate to find Mcpd that corresponds to designated flux limit
    nrefine, nspan = 2, 100
    Mlim = np.zeros((len(Mpl), len(Rth)))
    flag = np.zeros_like(Mlim, dtype=bool)
    for ir in range(len(Rth)):
        for ip in range(len(Mpl)):
            Mlo0, Mhi0 = 0.01, 1000
            for iref in range(nrefine):
                # flux grid for mass grid
                MM = np.logspace(np.log10(Mlo0), np.log10(Mhi0), nspan)
                FF = np.zeros(nspan)
                for i in range(len(MM)):
                    FF[i] = CPD_model(Mpl=Mpl[ip], 
                                      Mdot=eps_Mdot * Mpl[ip] * 1e-6, 
                                      Mcpd=dtog * MM[i], Tirrs=Tirrs, 
                                      incl=incl, p=psl, dpc=dpc, kap=kap,
                                      alb=alb, rtrunc=Rth[ir], Mstar=Mstar, 
                                      apl=apl, rin=rin, HR=HR, nu=nu)
   
                # check if all values are below Flim
                if (np.all(FF < Flim)):
                    iref = nrefine-1
                    flag[ip, ir] = True
                else:
                    # make refinement
                    ixlo = np.where((FF - Flim) < 0, (FF - Flim), 
                                    -np.inf).argmax()
                    Mlo0, Mhi0 = MM[ixlo], MM[ixlo+1]

            # final refinement is a linear interpolation
            if flag[ip, ir]:
                Mlim[ip, ir] = np.nan
            else:
                mint = interp1d(FF, MM)
                Mlim[ip, ir] = mint(Flim)

    return Mlim
