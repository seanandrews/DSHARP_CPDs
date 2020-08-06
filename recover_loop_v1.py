import os, sys, time
import numpy as np
from astropy.io import fits
from razmap import razmap


# bookkeeping
base = 'SR4_continuum'

# load parameters file
Fstr, mstr, rstr, azstr = np.loadtxt(base+'_mpars.txt', dtype=str).T
Fcpd, mdl, rcpd, azcpd = np.loadtxt(base+'_mpars.txt').T

# fix stupid bookkeeping error
Fcpd[(Fcpd % 10) == 9] += 1

# fixed parameters
offRA, offDEC = -0.060, -0.509
incl, PA = 22., 26.

# bookkeeping
recov_file = 'SR4_continuum_CPDrecoveries.txt'
os.system('rm -rf '+recov_file)

# loop
for i in range(len(Fstr)):

    # load the residual image into a (r, az) map object
    rfile = base+'_F'+Fstr[i]+'uJy_'+mstr[i]+'.resid.fits'
    rbins = np.arange(0.003, 0.6, 0.003)
    tbins = np.linspace(-180, 180, 181)    
    raz_obj = razmap('resid_images/'+rfile, rbins, tbins, incl=incl, PA=PA, 
                     offx=offRA, offy=offDEC)    

    # extract the (r, az) map itself
    raz_d = raz_obj.raz_map

    # isolate the gap region (+/-15 mas from mean gap radius of 80 mas)
    r_wedge = ((rbins >= 0.065) & (rbins <= 0.095))
    raz_gap = raz_d[:, r_wedge]
    rbins_gap = rbins[r_wedge]

    # find the peak
    tpeak_idx, rpeak_idx = np.unravel_index(np.argsort(raz_gap, axis=None), 
                                            raz_gap.shape)
    az_peak, r_peak = tbins[tpeak_idx[-1]], rbins_gap[rpeak_idx[-1]]

    # peak brightness (in uJy)
    f_peak = 1e6 * raz_gap[tpeak_idx[-1], rpeak_idx[-1]]


    # local significance of peak
    # make a 2-D mask
    mask = np.zeros_like(raz_gap, dtype='bool')
    az_wid = 15.
    t_exc = ((tbins >= (az_peak - az_wid)) & (tbins <= (az_peak + az_wid)))
    mask[t_exc, :] = True

    # calculate noise outside the masked region
    noise_gap = 1e6 * np.std(raz_gap[~mask])

    # SNR of the peak
    SNR = f_peak / noise_gap


    # record the outcome
    with open(recov_file, 'a') as f:
        f.write('%.0f  %.0f  %s  %.3f  %.3f  %4i  %4i  %.3f\n' % \
                (Fcpd[i], f_peak, mstr[i], rcpd[i], r_peak, azcpd[i], az_peak,
                 SNR))


