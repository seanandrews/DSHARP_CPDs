import numpy as np

def inject_CPD(data, CPD_pars, incl=0, PA=0, offRA=0, offDEC=0):

    # parse input data
    u, v, vis, wgt = data

    # parse CPD parameters (flux in mJy, radius in arcsec, az in degrees)
    F_cpd, r_cpd, az_cpd = CPD_pars

    # disk geometry parameters to proper units
    inclr = np.radians(incl)
    PAr = np.radians(PA)
    offx = -offRA * np.pi / (180 * 3600)
    offy = -offDEC * np.pi / (180 * 3600)

    # CPD location in disk-frame cartesian coordinates
    x_cpd_disk = r_cpd * np.sin(0.5 * np.pi - np.radians(az_cpd))
    y_cpd_disk = r_cpd * np.cos(0.5 * np.pi - np.radians(az_cpd))

    # CPD location in sky-frame cartesian coordinates
    x_cpd_sky = x_cpd_disk * np.cos(PAr) * np.cos(inclr) + \
                y_cpd_disk * np.sin(PAr)
    y_cpd_sky = -x_cpd_disk * np.sin(PAr) * np.cos(inclr) + \
                y_cpd_disk * np.cos(PAr)

    # now convert these to offset coordinates in proper units
    off_cpd_x = -x_cpd_sky * np.pi / (180 * 3600)
    off_cpd_y = -y_cpd_sky * np.pi / (180 * 3600)

    # make a point source visibility model at the disk-plane origin
    vis_CPD = F_cpd * 1e-3 * np.ones_like(u) + 1.0j * np.zeros_like(u)

    # phase shift to the sky-plane position (if disk at phase center)
    cpd_shift = np.exp(-2 * np.pi * 1.0j * (u*off_cpd_x + v*off_cpd_y))
    vis_CPD *= cpd_shift

    # phase shift to the offset sky-plane position
    offxy_shift = np.exp(-2 * np.pi * 1.0j * (u*offx + v*offy))
    vis_CPD *= offxy_shift

    # return the visibilities with the injected CPD
    return (vis + vis_CPD)
