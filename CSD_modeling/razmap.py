import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d

def razmap(imfits, rbins, tbins, incl=0, PA=0, offx=0, offy=0):

    # load the FITS image and header
    dat = fits.open(imfits)
    img = np.squeeze(dat[0].data)
    hdr = dat[0].header

    # parse coordinates
    nx, ny = hdr['NAXIS1'], hdr['NAXIS2']
    RA = hdr['CRVAL1'] + hdr['CDELT1'] * (np.arange(nx) - (hdr['CRPIX1'] - 1))
    DEC = hdr['CRVAL2'] + hdr['CDELT2'] * (np.arange(ny) - (hdr['CRPIX2'] - 1))
    RAo = 3600 * (RA - hdr['CRVAL1']) - offx 
    DECo = 3600 * (DEC - hdr['CRVAL2']) - offy
    dRA, dDEC = np.meshgrid(RAo, DECo)

    # beam parameters
    bmaj, bmin, bPA = 3600 * hdr['BMAJ'], 3600 * hdr['BMIN'], hdr['BPA']

    # convert geometric parameters into radians
    inclr, PAr = np.radians(incl), np.radians(PA)

    # deproject and rotate into the disk frame coordinate system
    xd = (dRA * np.cos(PAr) - dDEC * np.sin(PAr)) / np.cos(inclr)
    yd = (dRA * np.sin(PAr) + dDEC * np.cos(PAr))

    # convert to polar coordinates (theta = 0 is minor axis, rotates clockwise 
    # in the sky-plane)
    r = np.sqrt(xd**2 + yd**2)
    theta = np.degrees(np.arctan2(yd, xd))

    # compute the radial and azimuthal bin widths
    dr = np.abs(rbins[1] - rbins[0])
    dt = np.abs(tbins[1] - tbins[0])

    # initialize the (r, az)-map and radial profile
    rtmap = np.empty((len(tbins), len(rbins)))
    SBr, err_SBr = np.empty(len(rbins)), np.empty(len(rbins))
    int_flags = np.zeros((len(tbins), len(rbins)))

    # loop through the bins to populate the (r, az)-map and radial profile
    for i in range(len(rbins)):
        # identify pixels that correspond to the radial bin (i.e., in this annulus)
        in_annulus = ((r >= rbins[i] - 0.5 * dr) & (r < (rbins[i] + 0.5 * dr)))

        # accumulate the azimuth values and intensities in this annulus
        az_annulus = theta[in_annulus]
        SB_annulus = img[in_annulus]

        # compute the azimuthally-averaged brightness profile (and scatter)
        SBr[i], err_SBr[i] = np.average(SB_annulus), np.std(SB_annulus)

        # populate the azimuthal bins for the (r, az)-map at this radius
        for j in range(len(tbins)):
            # identify pixels that correspond to the azimuthal bin
            in_wedge = ((az_annulus >= (tbins[j] - 0.5 * dt)) & 
                        (az_annulus < (tbins[j] + 0.5 * dt)))

            # if there's pixels in that bin, average the intensities
            if (len(SB_annulus[in_wedge]) > 0):
                rtmap[j, i] = np.average(SB_annulus[in_wedge])
            else:
                rtmap[j,i] = -1e10    # temporary -- we fix this below
                int_flags[j,i] = 1

    # now "fix" the (r, az)-map where there are too few pixels in certain az bins
    # (especially relevant at low r values)
    for i in range(len(rbins)):
        # extract an azimuthal slice of the (r, az)-map
        az_slice = rtmap[:,i]

        # identify if there's missing information in an az bin along that slice:
        # if so, fill it in with linear interpolation along the slice
        if np.any(az_slice < -1e5):
            # extract non-problematic bins in the slice
            xslice, yslice = tbins[az_slice >= -1e5], az_slice[az_slice >= -1e5]
        
            # pad the arrays to make sure they span a full circle in azimuth
            xslice_ext = np.pad(xslice, 1, mode='wrap')
            xslice_ext[0] -= 360.
            xslice_ext[-1] += 360.
            yslice_ext = np.pad(yslice, 1, mode='wrap')

            # define the interpolation function
            raz_func = interp1d(xslice_ext, yslice_ext, bounds_error=True)

            # interpolate and replace those bins in the (r, az)-map
            fixed_slice = raz_func(tbins)
            rtmap[:,i] = fixed_slice

    class raz_out:
        def __init__(self, raz_map, raz_int_flags, r, az, prof, eprof, 
                     bmaj, bmin, bpa):
            self.raz_map = raz_map
            self.raz_int_flags = raz_int_flags
            self.r = r
            self.az = az
            self.prof = prof
            self.eprof = eprof
            self.bmaj = bmaj
            self.bmin = bmin
            self.bpa = bpa

    return raz_out(rtmap, int_flags, rbins, tbins, SBr, err_SBr, bmaj, bmin, bPA)
