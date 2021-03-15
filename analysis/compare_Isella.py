import os, sys
import numpy as np
from CPD_model import CPD_model


# Isella et al. 2019 example for PDS 70c
F = CPD_model(Mpl=4.0, Mdot=1e-8, Mcpd=0.0043, Tirrs=20., incl=51.7, kap=3.5,
              p=1., dpc=113.4, Mstar=0.76, apl=34.5, plot_struct=True,
              Rout=1.25, HR=0.1, Lpl=1.6e-4, Rpl=3.0, nu=350.6)
print(F)
os.system('mv CPD_model.structure.png PDS70c_model.structure.png')


# Isella et al. 2014 example for LkCa 15b
Rpl_hot, Teff_hot = 1.7, 2500
Lpl_hot = 4 * np.pi * 5.67e-5 * (Rpl_hot*6.9911e9)**2 * Teff_hot**4 / 3.282e33
Mcpd_hot = 0.17 * 1.898e33 / 5.974e27

F = CPD_model(Mpl=10., Mdot=1e-4, Mcpd=Mcpd_hot, Tirrs=40, incl=45, kap=0.002,
              p=0.75, dpc=150., Mstar=1.0, apl=16.0, plot_struct=True,
              Rout=1.4, HR=0.1, Lpl=Lpl_hot, Rpl=Rpl_hot, nu=43.)
print(F / 3.6)
os.system('mv CPD_model.structure.png LkCa15b_hot_model.structure.png')
