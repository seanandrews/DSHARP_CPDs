import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

RMS = 10.
Fcpd = 20.

Ndraws = 1000000000

# generate random draws of "noise"
pnoise = np.random.normal(0, RMS, Ndraws)

# generate random draws of noisy CPD
cpd = np.random.normal(Fcpd, RMS, Ndraws)

# compute the number of random draws where noise > cpd
dif = pnoise - cpd
Npeak_is_noise = 1.*len(dif[dif > 0])
print(Npeak_is_noise / Ndraws)

print(1. - 0.5 * (1 + erf(Fcpd / (2 * RMS))))
