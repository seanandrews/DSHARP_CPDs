import numpy as np

# define a function to generate a 1-D visibility "profile"
def deproject_vis(data, bins=np.array([0]), incl=0., PA=0., offx=0., offy=0.):
        
    # convert keywords into relevant units
    inclr = np.radians(incl)
    PAr = 0.5 * np.pi - np.radians(PA)
    offx *= -np.pi / (180 * 3600)
    offy *= -np.pi / (180 * 3600)

    # change to a deprojected, rotated coordinate system
    uprime = (data.u * np.cos(PAr) + data.v * np.sin(PAr))
    vprime = (-data.u * np.sin(PAr) + data.v * np.cos(PAr)) * np.cos(inclr)
    rhop = np.sqrt(uprime**2 + vprime**2)

    # phase shifts to account for offsets
    shifts = np.exp(-2 * np.pi * 1.0j * (data.u*-offx + data.v*-offy))
    visp = data.vis * shifts
    realp = visp.real
    imagp = visp.imag
    
    # create a class to return outputs
    class Vis_profile:
        def __init__(self, vis_prof, rho_uv, err_std, err_scat, nperbin):
            self.vis_prof = vis_prof 
            self.rho_uv = rho_uv 
            self.err_std = err_std
            self.err_scat = err_scat
            self.nperbin = nperbin

    # if requested, return a binned (averaged) representation
    wgt = data.wgt
    if (bins.size > 1):
        avbins = 1e3 * bins       # scale to lambda units (input in klambda)
        bwid = 0.5 * (avbins[1] - avbins[0])
        bvis = np.zeros_like(avbins, dtype='complex')
        berr_std = np.zeros_like(avbins, dtype='complex')
        berr_scat = np.zeros_like(avbins, dtype='complex')
        n_in_bin = np.zeros_like(avbins, dtype='int')
        for ib in np.arange(len(avbins)):
            inb = np.where((rhop >= avbins[ib] - bwid) & (rhop < avbins[ib] + bwid))
            if (len(inb[0]) >= 5):
                bRe, eRemu = np.average(realp[inb], weights=wgt[inb], returned=True)
                eRese = np.std(realp[inb])
                bIm, eImmu = np.average(imagp[inb], weights=wgt[inb], returned=True)
                eImse = np.std(imagp[inb])
                bvis[ib] = bRe + 1j*bIm
                berr_scat[ib] = eRese + 1j*eImse
                berr_std[ib] = 1 / np.sqrt(eRemu) + 1j / np.sqrt(eImmu)
                n_in_bin[ib] = np.size(bRe)
            else:
                bvis[ib] = 0 + 1j*0
                berr_scat[ib] = 0 + 1j*0
                berr_std[ib] = 0 + 1j*0
                n_in_bin[ib] = 0
        parser = np.where(berr_std.real != 0)
        output = Vis_profile(bvis[parser], avbins[parser], berr_std[parser], 
                             berr_scat[parser], n_in_bin[parser])
        return output
    
    # if not, returned the unbinned representation
    output = Vis_profile(realp + 1j*imagp, rhop, 1 / np.sqrt(wgt), 
                         1 / np.sqrt(wgt), np.zeros_like(rhop))

    return output
