import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.io import fits
sys.path.append('../')
import diskdictionary as disk

# target disk/gap; iteration
target = 'Elias20'
gap = 0
ix = '0'

# decisions!  How many pixels tolerated?  How many x astrometry RMS?
npix_tol = 2
ast_tol = 2



# load the recoveries file data
rec_file = target + '_gap' + str(gap) + '_recoveries.' + ix + '.txt'
Fi, Fr, mdl, ri, rr, azi, azr, xr, yr = np.loadtxt('recoveries/' + rec_file).T

# convert the injected astrometry from disk-polar to sky-Cartesian
azir = np.radians(azi)
inclr = np.radians(disk.disk[target]['incl'])
PAr = np.radians(disk.disk[target]['PA'])
xi = ri * np.sin(azir) * np.sin(PAr) + \
     ri * np.cos(azir) * np.cos(inclr) * np.cos(PAr)
yi = ri * np.sin(azir) * np.cos(PAr) - \
     ri * np.cos(azir) * np.cos(inclr) * np.sin(PAr)

# calculate the distance between injection and recovery signals
d_ir = np.sqrt((xi - xr)**2 + (yi - yr)**2)

# load beam parameters from residual image header
if np.logical_or((target == 'HD143006'), (target == 'HD163296')):
    imfile = '../CSD_modeling/data/'+target+'_resid_symm.JvMcorr.fits'
else:
    imfile = '../CSD_modeling/data/'+target+'_resid.JvMcorr.fits'
hd = fits.open(imfile)[0].header
beam_fwhm = np.sqrt(3600**2 * hd['BMAJ'] * hd['BMIN'])

# find maximum baseline length in kilometers
vdat = np.load('data/'+target+'_data.vis.npz')
freq, wave = hd['CRVAL3'], 2.99792e5 / hd['CRVAL3']
Bmax_km = (wave * np.sqrt(vdat['u']**2 + vdat['v']**2)).max()

# define std dev of astrometric precision
# Option A = astrometric uncertainty from Reid et al. 1988, ApJ, 330, 809
# Option B = astrometric uncertainty from Cycle 7 Tech. Handbook, p154
sigma_A = (4 / np.pi)**0.25 * beam_fwhm * disk.disk[target]['RMS'] / \
          (np.sqrt(8 * np.log(2)) * Fr)
sigma_B = 70 / ( (freq / 1e9) * Bmax_km * Fr / disk.disk[target]['RMS'] )
pix_size = np.abs(3600 * hd['CDELT2'])

# recovery criterion
dlim_A = np.maximum(ast_tol * sigma_A, npix_tol * pix_size * np.ones_like(Fi))
is_recovered_A = (d_ir <= dlim_A)
dlim_B = np.maximum(ast_tol * sigma_B, npix_tol * pix_size * np.ones_like(Fi))
is_recovered_B = (d_ir <= dlim_B)

# calculate recovery fractions (and Poisson uncertainties) at each unique Fcpd
Fcpd = np.unique(Fi)
frec_A, efrec_A = np.zeros_like(Fcpd), np.zeros_like(Fcpd)
frec_B, efrec_B = np.zeros_like(Fcpd), np.zeros_like(Fcpd)
for i in range(len(Fcpd)):
    in_f = (Fi == Fcpd[i])
    frec_A[i]  = 1.*len(Fi[is_recovered_A & in_f]) / len(Fi[in_f])
    efrec_A[i] = np.sqrt(len(Fi[is_recovered_A & in_f])) / len(Fi[in_f])
    frec_B[i]  = 1.*len(Fi[is_recovered_B & in_f]) / len(Fi[in_f])
    efrec_B[i] = np.sqrt(len(Fi[is_recovered_B & in_f])) / len(Fi[in_f])

# save the profiles in ASCII files
np.savetxt('recoveries/'+target+'_gap'+str(gap)+'_rprofs.'+ix+'.txt', 
           list(zip(Fcpd, frec_A, efrec_A, frec_B, efrec_B)), fmt='%.4f')




### RECOVERED FLUX DENSITY
from scipy.special import erf

def mk_cdf(x):
    return sorted(x), np.arange(len(x)) / len(x)

fig, ax = plt.subplots()
ix = np.where(frec_B >= 0.90)[0]
for i in ix:	#[10, 12, 14, 16, 18, 20, 22, 24]: #range(len(Fcpd)):
    # calculate and plot CDF of the successfully recovered mock CPD fluxes
    cdfx, cdfy = mk_cdf(Fr[(Fi == Fcpd[i]) & is_recovered_B])
    ax.plot(cdfx, cdfy, lw=2)

    # calculate a Gaussian CDF with a biased mean and some width
    Fbias, sigF = 25., 1.7*disk.disk[target]['RMS']
    Fbias, sigF = 35., 3.0*disk.disk[target]['RMS']
    mdl_cdf = 0.5 * (1 + erf((cdfx - (Fcpd[i] - Fbias)) / (np.sqrt(2) * sigF)))
    ax.plot(cdfx, mdl_cdf, ':k')
 

    ax.set_xlim([50, 300])
    ax.set_ylim([0, 1])

fig.savefig('assess_figs/'+target+'_Fcdf.png')
fig.clf()


# measure bias
fig, ax = plt.subplots()
resid_F = Fr - Fi
ax.plot(Fi[is_recovered_B], resid_F[is_recovered_B], ',k', rasterized=True)
mu_bias, sig_bias = np.zeros_like(Fcpd), np.zeros_like(Fcpd)
for i in range(len(Fcpd)):
    mu_bias[i]  = np.average(resid_F[(Fi == Fcpd[i]) & is_recovered_B])
    sig_bias[i] = np.std(resid_F[(Fi == Fcpd[i]) & is_recovered_B])
ax.errorbar(Fcpd, mu_bias, yerr=sig_bias, color='r', ls='none',
            marker='o', capsize=0.0)
print(np.mean(mu_bias[ix]), np.mean(sig_bias[ix]))
    
ax.set_xlim([0, 260])
ax.set_ylim([-100, 100])

fig.savefig('assess_figs/'+target+'_Fbias.png')
fig.clf()


sys.exit()


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



