import numpy as np
import scipy.constants as sc
from scipy.special import erf
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Gaussian2DKernel


class simple_disk:
    """
    Args:
        # Geometric Parameters
        inc (float): Inclination of the source in [degrees].
        PA (float): Position angle of the source in [degrees].
        x0 (Optional[float]): Source center offset along x-axis in [arcsec].
        y0 (Optional[float]): Source center offset along y-axis in [arcsec].
        dist (Optional[float]): Distance to the source in [pc].
        mstar (Optional[float]): Mass of the central star in [Msun].
        r_min (Optional[float]): Inner radius in [au].
        r_max (Optional[float]): Outer radius in [au].
        r0 (Optional[float]): Normalization radius in [au]. (r0 must be < r_l)
        r_l (Optional[float]): Turn-over radius in [au].
        z0 (Optional[float]): Emission height in [au] at r0.
        zpsi (Optional[float]): Index of z_l profile for r < r_l.
        zphi (Optional[float]): Exponential taper index of z_l profile at 
            r > r_l.

        # Brightness Temperatures
        Tb0 (Optional[float]): Brightness temperature in [K] at r0.
        Tbq (Optional[float]): Index of Tb profile for r < r_l.
        Tbeps (Optional[float]): Exponential taper index of Tb profile for 
            r > r_l.
        Tbmax (Optional[float]): Maximum Tb in [K].
        Tbmax_b (Optional[float]): Maximum Tb for back side of disk in [K].

        # Optical depth of front-side
        tau0 (Optional[float]): Optical depth at r0.
        tauq (Optional[float]): Index of optical depth profile for r < r_l
        taueta (Optional[float]): Exponential taper index for optical depth
            profile at r > r_l.
        taumax (Optional[float]): Maximum optical depth.

        # Line-widths
        dV0 (Optional[float]): Doppler line width in [m/s] at r0.
        dVq (Optional[float]): Index of line-width profile.
        dVmax (Optional[float]): Maximum line-width.
        xi_nt (Optional[float]): Non-thermal line-width fraction (of sound 
	    speed for the gas); can use if dV0, dVq are None.

        # Observational Parameters
        FOV (Optional[float]): Field of view of the model in [arcsec].
        Npix (Optional[int]): Number of pixels along each axis.
        mu_l (Optional[float]): Mean atomic weight for line of interest.

    """

    # Establish constants
    mu = 2.37
    msun = 1.98847e30
    mH = sc.m_p + sc.m_e

    # Establish useful conversion factors
    fwhm = 2.*np.sqrt(2.*np.log(2.))
    nwrap = 3


    def __init__(self, inc, PA, x0=0., y0=0., dist=100., mstar=1., 
                 r_min=0., r_max=1000., r0=10., r_l=100., 
                 z0=0., zpsi=1., zphi=np.inf,
                 Tb0=50., Tbq=0.5, Tbeps=np.inf, Tbmax=1000., Tbmax_b=20., 
                 tau0=100., tauq=0., taueta=np.inf, taumax=None,
                 dV0=None, dVq=None, dVmax=1000., xi_nt=0.,
                 FOV=None, Npix=128, mu_l=28):



        # Set the disk geometrical properties.
        self.x0, self.y0, self.inc, self.PA, self.dist = x0, y0, inc, PA, dist
        self.z0, self.zpsi, self.zphi = z0, zpsi, zphi
        self.r_l, self.r0, self.r_min, self.r_max = r_l, r0, r_min, r_max

        # Define the velocity, brightness and linewidth radial profiles.
        self.Tb0, self.Tbq, self.Tbeps = Tb0, Tbq, Tbeps
        self.Tbmax, self.Tbmax_b = Tbmax, Tbmax_b

        # Set the observational parameters.
        self.FOV = 2.2 * self.r_max / self.dist if FOV is None else FOV
        self.Npix = Npix
        self.mu_l = mu_l


        # Build the disk model.
        self._populate_coordinates()
        self._set_brightness()


    # -- Model Building Functions -- #

    def _populate_coordinates(self):
        """
        Populate the coordinates needed for the model.
        """

        # Set sky cartesian coordinates, representing the pixels in the image.

        self.x_sky = np.linspace(-self.FOV / 2.0, self.FOV / 2.0, self.Npix)
        self.cell_sky = np.diff(self.x_sky).mean()
        self.x_sky, self.y_sky = np.meshgrid(self.x_sky, self.x_sky)

        # Use these pixels to define face-down disk-centric coordinates.

        self.x_disk = self.x_sky * self.dist
        self.y_disk = self.y_sky * self.dist
        self.cell_disk = np.diff(self.x_disk).mean()

        # Define three sets of cylindrical coordintes, the two emission
        # surfaces and the midplane. If `z0 = 0.0` then the two emission
        # surfaces are equal.

        self.r_disk = np.hypot(self.y_disk, self.x_disk)
        self.t_disk = np.arctan2(self.y_disk, self.x_disk)

        f = self.disk_coords(x0=self.x0, y0=self.y0, inc=self.inc, PA=self.PA,
                             z0=self.z0, zpsi=self.zpsi, zphi=self.zphi)

        self.r_sky_f = f[0] * self.dist
        self.t_sky_f = f[1]
        self.z_sky_f = f[2] * self.dist

        if self.z0 != 0.0:
            self._flat_disk = False
            b = self.disk_coords(x0=self.x0, y0=self.y0, inc=-self.inc,
                                 PA=self.PA, z0=self.z0, zpsi=self.zpsi,
                                 zphi=self.zphi)
        else:
            self._flat_disk = True
            b = f

        self.r_sky_b = b[0] * self.dist
        self.t_sky_b = b[1]
        self.z_sky_b = b[2] * self.dist

        # Define masks noting where the disk extends to.

        self._in_disk_f = np.logical_and(self.r_sky_f >= self.r_min,
                                         self.r_sky_f <= self.r_max)
        self._in_disk_b = np.logical_and(self.r_sky_b >= self.r_min,
                                         self.r_sky_b <= self.r_max)
        self._in_disk = np.logical_and(self.r_disk >= self.r_min,
                                       self.r_disk <= self.r_max)

    @property
    def r_sky(self):
        return self.r_sky_f

    @property
    def t_sky(self):
        return self.t_sky_f

    def _set_brightness(self):
        """
        Sets the brightness profile in [K].
        """
        self.Tb_f = self.Tb0 * (self.r_sky_f / self.r0)**(-self.Tbq) * \
                    np.exp(-(self.r_sky_f / self.r_l)**self.Tbeps)
        self.Tb_f = self.Tb0 * (self.r_sky_f / self.r0)**(-self.Tbq)
        Rtrans = 90.
        Tb_trans = self.Tb0 * (Rtrans / self.r0)**(-self.Tbq)
        outer = (self.r_sky_f >= Rtrans)
        self.Tb_f[outer] = Tb_trans * (self.r_sky_f[outer] / Rtrans)**(-5.)
        gap1 = (self.r_sky_f >= 15.) & (self.r_sky_f <= 18.)
        gap2 = (self.r_sky_f >= 70.) & (self.r_sky_f <= 82.)
        self.Tb_f[gap1] *= 0.01
        self.Tb_f[gap2] *= 0.05
        self.Tb_f = np.clip(self.Tb_f, 0.0, self.Tbmax)
        self.Tb_f = np.where(self._in_disk_f, self.Tb_f, 0.0)
        if self._flat_disk:
            self.Tb_b = None
        else:
            self.Tb_b = self.Tb0 * (self.r_sky_f / self.r0)**(-self.Tbq) * \
                        np.exp(-(self.r_sky_f / self.r_l)**self.Tbeps)
            self.Tb_b = np.clip(self.Tb_b, 0.0, self.Tbmax_b)
            self.Tb_b = np.where(self._in_disk_b, self.Tb_b, 0.0)

    def interpolate_model(self, x, y, param, x_unit='au', param_max=None,
                          interp1d_kw=None):
        """
        Interpolate a user-provided model for the brightness temperature
        profile or the line width.

        Args:
            x (array): Array of radii at which the model is sampled at in units
                given by ``x_units``, either ``'au'`` or ``'arcsec'``.
            y (array): Array of model values evaluated at ``x``. If brightness
                temperature, in units of [K], or for linewidth, units of [m/s].
            param (str): Parameter of the model, either ``'Tb'`` for brightness
                temperature, or ``'dV'`` for linewidth.
            x_unit (Optional[str]): Unit of the ``x`` array, either
                ``'au'`` or ``'arcsec'``.
            param_max (Optional[float]): If provided, use as the maximum value
                for the provided parameter (overwriting previous values).
            interp1d_kw (Optional[dict]): Dictionary of kwargs to pass to
                ``scipy.interpolate.intep1d`` used for the linear
                interpolation.
        """
        from scipy.interpolate import interp1d

        # Value the input models.

        if x.size != y.size:
            raise ValueError("`x.size` does not equal `y.size`.")
        if x_unit.lower() == 'arcsec':
            x *= self.dist
        elif x_unit.lower() != 'au':
            raise ValueError("Unknown `radii_unit` {}.".format(x_unit))
        if y[0] != 0.0 or y[-1] != 0.0:
            print("First or last value of `y` is non-zero and may cause " +
                  "issues with extrapolated values.")

        # Validate the kwargs passed to interp1d.

        ik = {} if interp1d_kw is None else interp1d_kw
        ik['bounds_error'] = ik.pop('bounds_error', False)
        ik['fill_value'] = ik.pop('fill_value', 'extrapolate')
        ik['assume_sorted'] = ik.pop('assume_sorted', False)

        # Interpolate the functions onto the coordinate grids.

        if param.lower() == 'tb':
            self.Tb_f = interp1d(x, y, **ik)(self.r_sky_f)
            self.Tb_f = np.clip(self.Tb_f, 0.0, param_max)
            if self.r_sky_b is not None:
                self.Tb_b = interp1d(x, y, **ik)(self.r_sky_b)
                self.Tb_b = np.clip(self.Tb_b, 0.0, param_max)
            self.Tb0, self.Tbq, self.Tbmax = np.nan, np.nan, param_max

        else:
            raise ValueError("Unknown 'param' value {}.".format(param))

    @property
    def Tb_disk(self):
        """
        Disk-frame brightness profile.
        """
        Tb = self.Tb0 * (self.r_sky_f / self.r0)**(-self.Tbq) * \
             np.exp(-(self.r_sky_f / self.r_l)**self.Tbeps)
        return np.where(self._in_disk, Tb, np.nan)

    # -- Deprojection Functions -- #

    def disk_coords(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=0.0, zpsi=0.0,
                    zphi=0.0, frame='cylindrical'):
        r"""
        Get the disk coordinates given certain geometrical parameters and an
        emission surface. The emission surface is parameterized as a powerlaw
        profile:

        .. math::

            z(r) = z_0 \times \left(\frac{r}{1^{\prime\prime}}\right)^{\psi} +
            z_1 \times \left(\frac{r}{1^{\prime\prime}}\right)^{\varphi}

        Where both ``z0`` and ``z1`` are given in [arcsec]. For a razor thin
        disk, ``z0=0.0``, while for a conical disk, ``psi=1.0``. Typically
        ``z1`` is not needed unless the data is exceptionally high SNR and well
        spatially resolved.

        It is also possible to override this parameterization and directly
        provide a user-defined ``z_func``. This allow for highly complex
        surfaces to be included. If this is provided, the other height
        parameters are ignored.

        Args:
            x0 (Optional[float]): Source right ascension offset [arcsec].
            y0 (Optional[float]): Source declination offset [arcsec].
            inc (Optional[float]): Source inclination [deg].
            PA (Optional[float]): Source position angle [deg]. Measured
                between north and the red-shifted semi-major axis in an
                easterly direction.
            z0 (Optional[float]): Aspect ratio at 1" for the emission surface.
                To get the far side of the disk, make this number negative.
            psi (Optional[float]): Flaring angle for the emission surface.
            z1 (Optional[float]): Aspect ratio correction term at 1" for the
                emission surface. Should be opposite sign to ``z0``.
            phi (Optional[float]): Flaring angle correction term for the
                emission surface.
            frame (Optional[str]): Frame of reference for the returned
                coordinates. Either ``'polar'`` or ``'cartesian'``.

        Returns:
            Three coordinate arrays, either the cylindrical coordaintes,
            ``(r, theta, z)`` or cartestian coordinates, ``(x, y, z)``,
            depending on ``frame``.
        """

        # Check the input variables.

        frame = frame.lower()
        if frame not in ['cylindrical', 'cartesian']:
            raise ValueError("frame must be 'cylindrical' or 'cartesian'.")

        # Calculate the pixel values.

        r, t, z = self._get_flared_coords(x0, y0, inc, PA, self._z_func)
        if frame == 'cylindrical':
            return r, t, z
        return r * np.cos(t), r * np.sin(t), z

    def _z_func(self, r):
        """
        Returns the emission height in [arcsec].
        """
        z = self.z0 * (r * self.dist / self.r0)**self.zpsi * \
            np.exp(-(r * self.dist / self.r_l)**self.zphi) / self.dist
        return np.clip(z, 0., None)

    @staticmethod
    def _rotate_coords(x, y, PA):
        """
        Rotate (x, y) by PA [deg].
        """
        x_rot = y * np.cos(np.radians(PA)) + x * np.sin(np.radians(PA))
        y_rot = x * np.cos(np.radians(PA)) - y * np.sin(np.radians(PA))
        return x_rot, y_rot

    @staticmethod
    def _deproject_coords(x, y, inc):
        """
        Deproject (x, y) by inc [deg].
        """
        return x, y / np.cos(np.radians(inc))

    def _get_cart_sky_coords(self, x0, y0):
        """
        Return caresian sky coordinates in [arcsec, arcsec].
        """
        return self.x_sky - x0, self.y_sky - y0

    def _get_polar_sky_coords(self, x0, y0):
        """
        Return polar sky coordinates in [arcsec, radians].
        """
        x_sky, y_sky = self._get_cart_sky_coords(x0, y0)
        return np.hypot(y_sky, x_sky), np.arctan2(x_sky, y_sky)

    def _get_midplane_cart_coords(self, x0, y0, inc, PA):
        """
        Return cartesian coordaintes of midplane in [arcsec, arcsec].
        """
        x_sky, y_sky = self._get_cart_sky_coords(x0, y0)
        x_rot, y_rot = simple_disk._rotate_coords(x_sky, y_sky, PA)
        return simple_disk._deproject_coords(x_rot, y_rot, inc)

    def _get_midplane_polar_coords(self, x0, y0, inc, PA):
        """
        Return the polar coordinates of midplane in [arcsec, radians].
        """
        x_mid, y_mid = self._get_midplane_cart_coords(x0, y0, inc, PA)
        return np.hypot(y_mid, x_mid), np.arctan2(y_mid, x_mid)

    def _get_flared_coords(self, x0, y0, inc, PA, z_func):
        """
        Return cylindrical coordinates of surface in [arcsec, radians].
        """
        x_mid, y_mid = self._get_midplane_cart_coords(x0, y0, inc, PA)
        r_tmp, t_tmp = np.hypot(x_mid, y_mid), np.arctan2(y_mid, x_mid)
        for _ in range(5):
            y_tmp = y_mid + z_func(r_tmp) * np.tan(np.radians(inc))
            r_tmp = np.hypot(y_tmp, x_mid)
            t_tmp = np.arctan2(y_tmp, x_mid)
        return r_tmp, t_tmp, z_func(r_tmp)

    @property
    def xaxis_disk(self):
        """
        X-axis for the disk coordinates in [au].
        """
        return self.x_disk[0]

    @property
    def yaxis_disk(self):
        """
        y-axis for the disk coordinates in [au].
        """
        return self.y_disk[:, 0]

    @property
    def xaxis_sky(self):
        """
        X-axis for the sky coordinates in [arcsec].
        """
        return self.x_sky[0]

    @property
    def yaxis_sky(self):
        """
        Y-axis for the sky coordinates in [arcsec].
        """
        return self.y_sky[:, 0]

    # -- Helper Functions -- #

    def set_coordinates(self, x0=None, y0=None, inc=None, PA=None, dist=None,
                        z0=None, zpsi=None, r_min=None, r_max=None, FOV=None,
                        Npix=None):
        """
        Helper function to redefine the coordinate system.
        """
        self.x0 = self.x0 if x0 is None else x0
        self.y0 = self.y0 if y0 is None else y0
        self.inc = self.inc if inc is None else inc
        self.PA = self.PA if PA is None else PA
        self.dist = self.dist if dist is None else dist
        self.z0 = self.z0 if z0 is None else z0
        self.zpsi = self.zpsi if zpsi is None else zpsi
        self.r_min = self.r_min if r_min is None else r_min
        self.r_max = self.r_max if r_max is None else r_max
        self.FOV = self.FOV if FOV is None else FOV
        self.Npix = self.Npix if Npix is None else Npix
        self._populate_coordinates()
        self._set_brightness()

    def set_brightness(self, Tb0=None, Tbq=None, Tbmax=None, Tbmax_b=None):
        """
        Helper function to redefine the brightnes profile.
        """
        self.Tb0 = self.Tb0 if Tb0 is None else Tb0
        self.Tbq = self.Tbq if Tbq is None else Tbq
        self.Tbmax = self.Tbmax if Tbmax is None else Tbmax
        self.Tbmax_b = self.Tbmax_b if Tbmax_b is None else Tbmax_b
        self._set_brightness()

    # -- Pseudo Image Functions -- #

    def get_image(self):
        """
        Return the pseudo-cube with the given velocity axis.

        Args:
            velax (array): 1D array of channel centres in [m/s].
            dv0 (optional[ndarray]): An array of projected velocity
                perturbations in [m/s].
            bmaj (optional[float]): Synthesised beam major axis in [arcsec].
            bmin (optional[float]): Synthesised beam minor axis in [arcsec]. If
                only `bmaj` is specified, will assume a circular beam.
            bpa (optional[float]): Beam position angle in [deg].
            rms (optional[float]): RMS of the noise to add to the image.
            spectral_response (optional[list]): The kernel to convolve the cube
                with along the spectral dimension to simulation the spectral
                response of the telescope.

        Returns:
            cube (array): A 3D image cube.
        """

        # Calculate the flux from the front side of the disk.

        flux_f = self._calc_flux('f')

        # If `z0 != 0.0`, can combine the front and far sides based on a
        # two-slab approximation.

        if not self._flat_disk:
            flux_b = self._calc_flux('b')
            frac_f, frac_b = self._calc_frac()
            flux = frac_f * flux_f + frac_b * flux_b
        else:
            flux = flux_f

        # save the image
        image = flux
        return image 


    def _calc_flux(self, side='f'):
        """
        Calculate the emergent flux assuming single Gaussian component.
        """
        if side.lower() == 'f':
            Tb = self.Tb_f
        elif side.lower() == 'b':
            Tb = self.Tb_b
        else:
            quote = "Unknown 'side' value {}. Must be 'f' or 'r'."
            raise ValueError(quote.format(side))
        spec = np.empty_like(Tb)
        ok = (Tb > 0.)
        spec[~ok] = 0.
        spec[ok] = Tb[ok]
        return spec

    def _calc_frac(self):
        """
        Calculates the fraction of the front side of the disk realtive to the
        back side based on the optical depth.
        """
        return 1.0, 0.0

    def plot_brightness(self, fig=None, top_axis=True):
        """
        Plot the brightness temperature profile.
        """
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        x = self.r_sky_f.flatten()
        y = self.Tb_f.flatten()
        idxs = np.argsort(x)
        x, y = x[idxs], y[idxs]
        mask = np.logical_and(x >= self.r_min, x <= self.r_max)
        ax.plot(x[mask], y[mask])
        ax.set_xlabel('Radius [au]')
        ax.set_ylabel('BrightestTemperature [K]')
        if top_axis:
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim()[0] / self.dist,
                         ax.set_xlim()[1] / self.dist)
            ax2.set_xlabel('Radius [arcsec]')
        return fig


    def plot_emission_surface(self, fig=None, top_axis=True):
        """
        Plot the emission surface.
        """
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        x = self.r_sky_f.flatten()
        y = self._z_func(x / self.dist) * self.dist
        idxs = np.argsort(x)
        x, y = x[idxs], y[idxs]
        mask = np.logical_and(x >= self.r_min, x <= self.r_max)
        ax.plot(x[mask], y[mask])
        ax.set_xlabel('Radius [au]')
        ax.set_ylabel('Emission Height [au]')
        if top_axis:
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim()[0] / self.dist,
                         ax.set_xlim()[1] / self.dist)
            ax2.set_xlabel('Radius [arcsec]')
        return fig

    def plot_radii(self, ax, rvals, contour_kwargs=None, projection='sky',
                   side='f'):
        """
        Plot annular contours onto the axis.
        """
        contour_kwargs = {} if contour_kwargs is None else contour_kwargs
        contour_kwargs['colors'] = contour_kwargs.pop('colors', '0.6')
        contour_kwargs['linewidths'] = contour_kwargs.pop('linewidths', 0.5)
        contour_kwargs['linestyles'] = contour_kwargs.pop('linestyles', '--')
        if projection.lower() == 'sky':
            if 'f' in side:
                r = self.r_sky_f
            elif 'b' in side:
                r = self.r_sky_b
            else:
                raise ValueError("Unknown 'side' value {}.".format(side))
            x, y, z = self.x_sky[0], self.y_sky[:, 0], r / self.dist
        elif projection.lower() == 'disk':
            x, y, z = self.x_disk, self.y_disk, self.r_disk
        ax.contour(x, y, z, rvals, **contour_kwargs)

    @staticmethod
    def format_sky_plot(ax):
        """
        Default formatting for sky image.
        """
        from matplotlib.ticker import MaxNLocator
        ax.set_xlim(ax.get_xlim()[1], ax.get_xlim()[0])
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlabel('Offset [arcsec]')
        ax.set_ylabel('Offset [arcsec]')
        ax.scatter(0, 0, marker='x', color='0.7', lw=1.0, s=4)

    @staticmethod
    def format_disk_plot(ax):
        """
        Default formatting for disk image.
        """
        from matplotlib.ticker import MaxNLocator
        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1])
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlabel('Offset [au]')
        ax.set_ylabel('Offset [au]')
        ax.scatter(0, 0, marker='x', color='0.7', lw=1.0, s=4)

    @staticmethod
    def BuRd():
        """Blue-Red color map."""
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        c2 = plt.cm.Reds(np.linspace(0, 1, 32))
        c1 = plt.cm.Blues_r(np.linspace(0, 1, 32))
        colors = np.vstack((c1, np.ones(4), c2))
        return mcolors.LinearSegmentedColormap.from_list('BuRd', colors)

    @staticmethod
    def RdBu():
        """Red-Blue color map."""
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        c2 = plt.cm.Reds(np.linspace(0, 1, 32))
        c1 = plt.cm.Blues_r(np.linspace(0, 1, 32))
        colors = np.vstack((c1, np.ones(4), c2))[::-1]
        return mcolors.LinearSegmentedColormap.from_list('RdBu', colors)

    @property
    def extent_sky(self):
        return [self.x_sky[0, 0],
                self.x_sky[0, -1],
                self.y_sky[0, 0],
                self.y_sky[-1, 0]]

    @property
    def extent_disk(self):
        return [self.x_sky[0, 0] * self.dist,
                self.x_sky[0, -1] * self.dist,
                self.y_sky[0, 0] * self.dist,
                self.y_sky[-1, 0] * self.dist]
