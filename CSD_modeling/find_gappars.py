import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../')
import diskdictionary as disk

target = 'SR4'
plaw_range = [0.05, 0.2]
r0, sig, amp = 0.079, 0.010, 4.0
plaw_index, plaw_norm = -2.6, 0.15
low = 1e-3

#target = 'GWLup'
#plaw_range = [0.1, 0.4]
#r0, sig, amp = 0.485, 0.018, 1.3
#low = 1e-3

#target = 'RULup'
#plaw_range = [0.05, 0.35]
#r0, sig, amp = 0.179, 0.010, 0.23
#low = 2e-2

#target = 'Elias20'
#plaw_range = [0.05, 0.4]
#r0, sig, amp = 0.181, 0.011, 0.27
#low = 2e-2

target = 'Sz129'
plaw_range = [0.18, 0.30]
r0, sig, amp = 0.250, 0.020, 0.17
r0, sig, amp = 0.376, 0.020, 0.22
plaw_index, plaw_norm = -2.8, -0.04
low = 2e-2

#target = 'HD143006'
#plaw_range = [0.1, 0.4]
#r0, sig, amp = 0.140, 0.040, 1.5
#plaw_index, plaw_norm = -1.20, -0.3
#r0, sig, amp = 0.315, 0.026, 0.7
#plaw_index, plaw_norm = -2.2, -0.05
#low = 1e-3

#target = 'GWLup'
#plaw_range = [0.1, 0.4]
#r0, sig, amp = 0.485, 0.014, 1.45
#low = 2e-3

#target = 'Elias24'
#plaw_range = [0.05, 0.3]
#r0, sig, amp = 0.415, 0.046, 1.40
#low = 5e-3

#target = 'HD163296'
#plaw_range = [0.1, 0.4]
#r0, sig, amp = 0.49, 0.058, 2.0
#plaw_index, plaw_norm = -0.80, 0.13
#r0, sig, amp = 0.85, 0.063, 1.0
#plaw_index, plaw_norm = -0.70, 0.10
#low = 2e-3

#target = 'AS209'
#plaw_range = [0.1, 0.4]
#r0, sig, amp = 0.51, 0.043, 1.90
#low = 2e-3
#plaw_index, plaw_norm = -0.75, -0.3
#r0, sig, amp = 0.80, 0.082, 1.80
#plaw_index, plaw_norm = -0.60, -0.3

autofit = False


# plotting conventions
plt.style.use('classic')
fig, ax = plt.subplots()


# load frankenstein radial brightness profile fit
pfile = 'fits/' + target + '_frank_profile_fit.txt'
r_frank, Inu_frank, eInu_frank = np.loadtxt(pfile).T

# convert to Jy / arcsec**2
Inu_frank /= (3600 * 180 / np.pi)**2
Inu_frank[Inu_frank <= 0] = 1e-10
lInu = np.log10(Inu_frank)

# plot the radial profile
ax.plot(r_frank, Inu_frank, 'k', lw=2)

# fit a power-law profile over some radial range
if autofit:
    rr = (r_frank >= plaw_range[0]) & (r_frank <= plaw_range[1])
    pfit = np.polyfit(r_frank[rr], lInu[rr], 1)
    p = np.poly1d(pfit)
    plaw = p(r_frank)
else:
    plaw = plaw_norm + plaw_index * r_frank
ax.plot(r_frank, 10**plaw, 'g')

# draw a Gaussian on that background power-law profile
gap = amp * np.exp(-0.5 * (r_frank - r0)**2 / sig**2)
ax.plot(r_frank, 10**(plaw - gap), 'r', lw=2)
HW = 1.5 * 0.5 * sig * 2.355

# mark the nominal gap(s)
rgi, rgo = disk.disk[target]['rgapi'], disk.disk[target]['rgapo']
for ir in range(len(rgi)):
    ax.fill_between([rgi[ir], rgi[ir], rgo[ir], rgo[ir]],
                    [low, 1000, 1000, low], color='darkgray', zorder=0)
ax.plot([r0-HW, r0-HW], [low, 1000], ':r', lw=2)
ax.plot([r0+HW, r0+HW], [low, 1000], ':r', lw=2)
print(HW * disk.disk[target]['distance'])
print(2 * HW)

# mark the outer edge of the disk
ax.plot([disk.disk[target]['rout'], disk.disk[target]['rout']], 
        [low, 1000], '--', color='darkgray', zorder=1)

# limits and labeling
Rlims = [0, 1.1*disk.disk[target]['rout']]
Tlims = [low, 1.2 * Inu_frank.max()]
ax.text(0.95, 0.93, disk.disk[target]['label'], 
        transform=ax.transAxes, ha='right', va='top')
ax.set_xlim(Rlims)
ax.set_ylim(Tlims)
ax.set_yscale('log')
ax.set_xlabel('radius  ($^{\prime\prime}$)')
ax.set_ylabel('$I_\\nu$  (Jy arcsec$^{-2}$)')

fig.savefig('../figs/'+target+'_findgaps.pdf')
