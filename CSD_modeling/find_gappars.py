import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../')
import diskdictionary as disk

target = 'SR4'
r0, sig, amp = 0.079, 0.010, 30.0
AA, BB = 0.7, 0.7
low = 1e-3

#target = 'RULup'
#AA, BB = 1.05, 0.8
#r0, sig, amp = 0.179, 0.010, 2.0
#low = 2e-2

#target = 'Elias20'
#AA, BB = 0.88, 1.25
#r0, sig, amp = 0.181, 0.011, 1.8
#low = 2e-2

#target = 'Sz129'
#AA, BB = 0.48, 0.9
#r0, sig, amp = 0.243, 0.020, 1.3
#AA, BB = 7, 3.3
#r0, sig, amp = 0.376, 0.020, 1.8
#low = 2e-2

#target = 'HD143006'
#r0, sig, amp = 0.140, 0.035, 20
#AA, BB = 0.3, 0.5
#r0, sig, amp = 0.315, 0.026, 5
#low = 1e-3

#target = 'GWLup'
#AA, BB = 0.4, 1.25
#r0, sig, amp = 0.485, 0.014, 10.
#low = 2e-3

#target = 'Elias24'
#AA, BB = 0.9, 0.75
#r0, sig, amp = 0.42, 0.035, 40.
#low = 5e-3

#target = 'HD163296'
#AA, BB = 1.6, 0.7
#r0, sig, amp = 0.49, 0.040, 50.0
#AA, BB = 2.0, 0.9
#r0, sig, amp = 0.85, 0.041, 12.0
#low = 2e-3

#target = 'AS209'
#r0, sig, amp = 0.51, 0.032, 25
#low = 2e-3
#AA, BB = 0.6, 0.7
#r0, sig, amp = 0.80, 0.062, 30


# plotting conventions
plt.style.use('classic')
fig, ax = plt.subplots()


# load frankenstein radial brightness profile fit
pfile = 'fits/' + target + '_frank_profile_fit.txt'
r_frank, Inu_frank, eInu_frank = np.loadtxt(pfile).T

# convert to Jy / arcsec**2
Inu_frank /= (3600 * 180 / np.pi)**2
Inu_frank[Inu_frank <= 0] = 1e-10

# fit a power-law profile over some radial range
plaw = AA * (r_frank / 0.1)**-BB

# gap correction
#gap = (1 - amp * np.exp(-0.5 * (r_frank - r0)**2 / sig**2))
gap_ = (amp - 1.) * np.exp(-0.5 * (r_frank - r0)**2 / sig**2)

# plot the "data"
ax.plot(r_frank, Inu_frank)

# plot the model power-law profile
ax.plot(r_frank, plaw, 'g')

# plot the gap-corrected model profile
#ax.plot(r_frank, plaw * gap, 'r')
ax.plot(r_frank, plaw / (1 + gap_), 'r')

# limits and labeling
Rlims = [0, 1.5*disk.disk[target]['rout']]
Tlims = [low, 1.5 * Inu_frank.max()]
ax.set_xlim(Rlims)
ax.set_ylim(Tlims)
ax.set_yscale('log')

fig.savefig('../figs/'+target+'_findgaps.pdf')
