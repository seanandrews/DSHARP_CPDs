import os, sys
import numpy as np
from astropy.io import ascii
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from CPD_model import CPD_model
sys.path.append('../')
import diskdictionary as disk

plt.rcParams.update({'font.size': 8})

# set up plot
fig = plt.figure(figsize=(3.5, 2.977))
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])

top, bottom, left, right = 0.99, 0.12, 0.13, 0.87


# set up axes, labels
Llims = [0.002, 500]
Mlims = [2e-5, 5]
ax.set_xlim(Mlims)
ax.set_xscale('log')
ax.set_xticks([0.0001, 0.001, 0.01, 0.1, 1])
ax.set_xticklabels(['10$^{-4}$', '0.001', '0.01', '0.1', '1'])
ax.set_xlabel('host mass  (M$_\odot$)')
ax.set_ylim(Llims)
ax.set_yscale('log')
ax.set_yticks([0.01, 0.1, 1, 10, 100])
ax.set_yticklabels(['0.01', '0.1', '1', '10', '100'])
ax.set_ylabel('1.3 mm luminosity (mJy at 150 pc)')


# load the main database
db = ascii.read('/pool/asha1/COMPLETED/ARAA/temp.csv', format='csv', 
                fast_reader=True)

# baseline selections
base = ( (db['FL_MULT'] != 'J') & (db['FL_MULT'] != 'B') & 
         (db['FL_MULT'] != 'HJB') & (db['FL_MULT'] != 'CB') & 
         (db['FL_MULT'] != 'WJ') & (db['FL_MULT'] != 'HJ') & 
         (db['FL_MULT'] != 'T') & (db['SED'] != 'III') & (db['SED'] != 'I') & 
         (db['SED'] != 'DEBRIS') & (db['FL_logMs'] == 0) )

# constants
d_ref, nu_ref, alp = 150., 240., 2.3


# calculate luminosities, upper limits, masses + uncertainties
L7 = db['F_B7'] * (nu_ref / db['nu_B7'])**alp * (db['DPC'] / d_ref)**2
L6 = db['F_B6'] * (nu_ref / db['nu_B6'])**alp * (db['DPC'] / d_ref)**2
eL7 = np.sqrt( (nu_ref/db['nu_B7'])**(2.*alp) * \
               (db['eF_B7']**2 + (0.1*db['F_B7'])**2) * \
               (db['DPC']/d_ref)**4 + \
               ( ((nu_ref / db['nu_B7'])**alp * \
                  db['F_B7']*(2.*db['DPC']/d_ref**2) * \
                 0.5*(db['EDPC_H']+db['EDPC_L']) )**2 ) )
eL6 = np.sqrt( (nu_ref/db['nu_B6'])**(2.*alp) * \
               (db['eF_B6']**2 + (0.1*db['F_B6'])**2) * \
               (db['DPC']/d_ref)**4 + \
               ( ((nu_ref / db['nu_B6'])**alp * \
                  db['F_B6']*(2.*db['DPC']/d_ref**2) * \
                 0.5*(db['EDPC_H']+db['EDPC_L']) )**2 ) )
limL7 = db['LIM_B7'] * (nu_ref/db['nu_B7'])**alp * (db['DPC']/d_ref)**2
limL6 = db['LIM_B6'] * (nu_ref/db['nu_B6'])**alp * (db['DPC']/d_ref)**2
Mstar = 10.**(db['logMs'])
eM_hi = 10.**(db['elogMs_H']+db['logMs']) - Mstar
eM_lo = Mstar - 10.**(db['logMs']-db['elogMs_L'])


# targets with B6 detections
detB6 = ( (db['FL_B6'] == 0) & base )
L_detB6 = L6[detB6]
eL_detB6 = eL6[detB6]
M_detB6 = Mstar[detB6]
eMhi_detB6 = eM_hi[detB6]
eMlo_detB6 = eM_lo[detB6]

# targets with **only** B7 detections (i.e., no B6 or B6 limit)
detB7 = ( (db['FL_B6'] != 0) & (db['FL_B7'] == 0) & base )
L_detB7 = L7[detB7]
eL_detB7 = eL7[detB7]
M_detB7 = Mstar[detB7]
eMhi_detB7 = eM_hi[detB7]
eMlo_detB7 = eM_lo[detB7]

# targets with **only** limits or missing data
lims = ( (db['FL_B7'] != 0) & (db['FL_B6'] != 0) & base )
dlims = np.ma.column_stack( (limL7[lims], limL6[lims]) )
L_lims = np.ma.min(dlims, 1)
M_lims = Mstar[lims]
eMhi_lims = eM_hi[lims]
eMlo_lims = eM_lo[lims]

# combine all detections
Lmm = np.ma.concatenate( (L_detB6, L_detB7) )
eLmm = np.ma.concatenate( (eL_detB6, eL_detB7) )
Ms = np.ma.concatenate( (M_detB6, M_detB7) )
eMs_hi = np.ma.concatenate( (eMhi_detB6, eMhi_detB7) )
eMs_lo = np.ma.concatenate( (eMlo_detB6, eMlo_detB7) )

# plot
ax.errorbar(M_lims, L_lims, yerr=0.35*L_lims, uplims=True, marker='None',
            color='C0', alpha=0.3, capsize=1.5, linestyle='None')
ax.errorbar(M_lims, L_lims, xerr=[eMlo_lims, eMhi_lims], yerr=0.,
            marker='None', color='C0', alpha=0.3, linestyle='None')
ax.errorbar(Ms, Lmm, xerr=[eMs_lo, eMs_hi], yerr=eLmm, marker='o',
            color='C0', markersize=3., linestyle='None', elinewidth=1.0,
            alpha=0.65)

# simple model
mx = np.logspace(-4, 2, 1024)
my = 60. * mx**1.7
ax.plot(mx, my, '--k', lw=1.0, zorder=0)


# PMCs from Wu et al.
wu_mm  = np.array([0.019, 0.011, 0.098, -0.002, 0.017, 0.033])
wu_emm = np.array([0.052, 0.045, 0.071,  0.043, 0.041, 0.039])
wu_ms  = np.array([   17.,    12.,    18.,      9.,    14.,    20.])
wu_ems = np.array([    5.,     2.,     3.,      2.,     3.,     4.])
wu_dpc = np.array([  191,   139,   137,    144,   135,   131])
nu_wu = (333.8 + 335.75 + 347.75) / 3.

wu_Lmm = (wu_mm + 3*wu_emm) * (nu_ref / nu_wu)**alp * (wu_dpc / d_ref)**2

wu_ms *= (1.898e30 / 1.989e33)
wu_ems *= (1.898e30 / 1.989e33)
ax.errorbar(wu_ms, wu_Lmm, yerr=0.35*wu_Lmm, uplims=True, marker='None',
            color='C1', capsize=1.5, linestyle='None', alpha=0.6)
ax.errorbar(wu_ms, wu_Lmm, xerr=wu_ems, yerr=0., marker='None', color='C1',
            linestyle='None', elinewidth=2)

# GQ Lup b
mac_mm = np.array([0.15])
mac_ms = np.array([23.])
mac_ems = np.array([13.])
mac_dpc = np.array([156.3])
nu_mac = np.array([338.])
mac_Lmm = mac_mm * (nu_ref / nu_mac)**alp * (mac_dpc / d_ref)**2

mac_ms *= (1.898e30 / 1.989e33)
mac_ems *= (1.898e30 / 1.989e33)
ax.errorbar(mac_ms, mac_Lmm, yerr=0.35*mac_Lmm, uplims=True, marker='None',
            color='C1', capsize=1.5, linestyle='None', alpha=0.6)
ax.errorbar(mac_ms, mac_Lmm, xerr=mac_ems, yerr=0., marker='None', color='C1',
            linestyle='None', elinewidth=2)

# 2MASS 1207 A
tmA_mm = np.array([0.620])
tmA_emm = np.array([0.067])
tmA_ms = np.array([25.]) * 1.898e30 / 1.989e33
tmA_ems = np.array([13.]) * 1.898e30 / 1.989e33
tmA_dpc = np.array([52.8])
nu_tmA = np.array([338.])
tmA_Lmm = tmA_mm * (nu_ref / nu_tmA)**alp * (tmA_dpc / d_ref)**2
tmA_eLmm = tmA_emm * (nu_ref / nu_tmA)**alp * (tmA_dpc / d_ref)**2

ax.errorbar(tmA_ms, tmA_Lmm, xerr=tmA_ems, yerr=tmA_eLmm, marker='d',
            color='C3', markersize=5., linestyle='None', elinewidth=1.5)

# 2MASS 1207 b
tmB_mm = np.array([0.078])
tmB_ms = np.array([5.]) * 1.898e30 / 1.989e33
tmB_ems = np.array([2.]) * 1.898e30 / 1.989e33
tmB_dpc = np.array([52.8])
nu_tmB = np.array([338.])
tmB_Lmm = tmB_mm * (nu_ref / nu_tmB)**alp * (tmB_dpc / d_ref)**2

ax.errorbar(tmB_ms, tmB_Lmm, yerr=0.35*tmB_Lmm, uplims=True, marker='None',
            color='C1', capsize=1.5, linestyle='None', alpha=0.6)
ax.errorbar(tmB_ms, tmB_Lmm, xerr=tmB_ems, yerr=0., marker='None', color='C1',
            linestyle='None', elinewidth=2)



# PDS 70c from Isella et al.
pds_mm  = np.array([0.106])
pds_emm = np.array([0.019])
pds_ms  = np.array([8.]) * (1.898e30 / 1.989e33)
pds_ems = np.array([4.]) * (1.898e30 / 1.989e33)
nu_pds = 2.9979e14 / 855. / 1e9

pds_Lmm = pds_mm * (nu_ref / nu_pds)**alp * (113.4 / d_ref)**2
pds_eLmm = pds_emm * (nu_ref / nu_pds)**alp * (113.4 / d_ref)**2

ax.errorbar(pds_ms, pds_Lmm, xerr=pds_ems, yerr=pds_eLmm, marker='o',
            color='C3', markersize=5., linestyle='None', elinewidth=1.5)


# DSHARP CPDs from this paper
dsh_mm   = np.array([0.065, 0.055, 0.046, 0.070, 0.070, 0.118, 0.104, 0.123, 0.055])
dsh_ms   = np.array([ 0.65,  0.84,  0.03, 19.91, 0.33, 2.18, 0.14, 2.16, 0.05])
dsh_hems = np.array([ 0.14, 0.16, 0.14, 0.16, 0.16, 0.16, 0.16, 0.13, 0.2])
dsh_lems = np.array([ 0.17, 0.14, 0.17, 0.14, 0.14, 0.14, 0.14, 0.16, 0.2])
# name             AS209_1, Elias24_1, GWLup_1, HD143006_0, HD143006_1,
#                  HD163296_0, HD163296_1, SR 4, Elias 20
nu_dsh = np.array([239.0, 231.9, 239.0, 239.0, 239.0, 239.0, 239.0, 239.0, 239.0])
ddsh = np.array([121.2, 139.3, 155.2, 167.3, 167.3, 101.0, 101.0, 134.8, 137.5])

ms_lo = 10**(np.log10(dsh_ms) - dsh_lems) * (1.898e30 / 1.989e33)
ms_hi = 10**(np.log10(dsh_ms) + dsh_hems) * (1.898e30 / 1.989e33)
dsh_Lmm = dsh_mm * (nu_ref / nu_dsh)**alp * (ddsh / d_ref)**2

print(ms_lo, ms_hi, dsh_Lmm)

#for i in range(len(dsh_Lmm)):
#    ax.plot([ms_lo[i], ms_hi[i]], [dsh_Lmm[i], dsh_Lmm[i]], 'C2', lw=2)

ax.errorbar(dsh_ms * 1.898e30 / 1.989e33, dsh_Lmm, yerr=0.35*dsh_Lmm, 
            uplims=True, marker='None', color='C2', capsize=1.5, 
            linestyle='None', alpha=0.6)
for i in range(len(dsh_Lmm)):
    ax.plot([ms_lo[i], ms_hi[i]], [dsh_Lmm[i], dsh_Lmm[i]], 'C2', lw=2)


# optically thick CPD models
mMpl = np.logspace(np.log10(0.02), np.log10(20.), 10)
eps_Mdot, mRth = 1, 0.3
kabs, alb = 2.4, 0.0

mFcpd_lo, mFcpd_hi = np.zeros_like(mMpl), np.zeros_like(mMpl)
for im in range(len(mMpl)):
    mFcpd_lo[im] = CPD_model(Mpl=mMpl[im], Mdot=mMpl[im] * 1e-6, 
                             Mcpd=1e8 * mMpl[im], Tirrs=27., incl=0., p=0.75, 
                             dpc=150., kap=kabs, alb=alb, rtrunc=mRth, 
                             Mstar=1.0, apl=53., nu=240)
    mFcpd_hi[im] = CPD_model(Mpl=mMpl[im], Mdot=100 * mMpl[im] * 1e-6,
                             Mcpd=1e8 * mMpl[im], Tirrs=27., incl=0., p=0.75,
                             dpc=150., kap=kabs, alb=alb, rtrunc=mRth,
                             Mstar=1.0, apl=53., nu=240)

ax.plot(mMpl * 1.898e30 / 1.989e33, mFcpd_lo * 1e-3, ':', color='silver', lw=1.0, zorder=0)
ax.plot(mMpl * 1.898e30 / 1.989e33, mFcpd_hi * 1e-3, ':', color='silver', lw=1.0, zorder=0)


# individualized
iFcpd = np.zeros_like(dsh_ms)
itargs = ['AS209', 'Elias24', 'GWLup', 'HD143006', 'HD143006', 'HD163296', 
          'HD163296', 'SR4', 'Elias20']
igaps = [1, 0, 0, 0, 1, 0, 1, 0, 0]
for i in range(len(dsh_ms)):
    Mpl = dsh_ms[i]
    dpc = disk.disk[itargs[i]]['distance']
    apl = disk.disk[itargs[i]]['rgap'][igaps[i]] * dpc
    lstar = disk.disk[itargs[i]]['lstar'] * 3.828e33
    Tirrs = (0.02 * lstar / (8 * np.pi * 5.67e-5 * (apl * 1.496e13)**2))**0.25
    incl = disk.disk[itargs[i]]['incl']
    print(disk.disk[itargs[i]]['mstar'])

    iFcpd[i] = CPD_model(Mpl=Mpl, Mdot=eps_Mdot * Mpl * 1e-6,
                         Mcpd=1e5 * Mpl, Tirrs=Tirrs, incl=incl, p=0.75,
                         dpc=dpc, kap=kabs, alb=alb, rtrunc=mRth, 
                         Mstar=disk.disk[itargs[i]]['mstar'], apl=apl, nu=240)
    iFcpd[i] *= 1e-3 * (dpc / 150.)**2


# annotations
ax.text(0.07, 100, 'stars', ha='right', va='center', color='C0')
ax.text(0.04, 0.06, 'PMCs', ha='left', va='center', color='C1')
ax.text(0.0003, 0.15, 'DSHARP', ha='center', va='center', color='C2')
ax.text(0.007, 0.017, 'PDS 70c', ha='right', va='center', color='C3',
        fontsize=6)
ax.text(0.03, 0.025, '2M1207A', ha='left', va='center', color='C3', fontsize=6)
ax.text(0.008, 0.004, '2M1207b', ha='left', va='center', color='C1', fontsize=6)



fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
fig.savefig('../figs/lmm_mstar.pdf')
fig.clf()
