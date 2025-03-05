# %%
import numpy as np
import xarray as xr
from numpy.typing import ArrayLike


class Age2d18Ow:
    """Convert age to d18O of seawater.

    The conversion is based on the sea level height at a given age derived by Spratt et al. [1]_.

    References
    ----------
    .. [1] Spratt, R. M. and Lisiecki, L. E.: A Late Pleistocene sea level stack, Clim. Past, 12, 1079–1092, https://doi.org/10.5194/cp-12-1079-2016, 2016.

    Examples
    --------

    Create a converter object and convert an age to d18Ow:

    >>> conv = Age2d18Ow()
    >>> conv.convert(0)
    0.0

    Convert multiple ages using a linear or nonlinear method:

    >>> ages = np.linspace(0, 50, 100)
    >>> d18Ow = conv.convert(ages, method="linear")

    """

    def __init__(self, random_generator: np.random.Generator | int | None = None):
        self.rng = np.random.default_rng(random_generator)

        self.sealevel = xr.open_dataset("data/sealevel/spratt2016_corrected.nc")

    def _sealevel_to_d18Ow_nonlinear(
        self,
        x: ArrayLike,
        dx: ArrayLike | None = None,
    ) -> ArrayLike:
        """Convert sealevel (x) to d18Ow (y) using a nonlinear relationship.

        REFERENCE ????

        Parameters
        ----------
        x : ArrayLike
            Sea level height in meters relative to today. Below present day sea level is negative.
        dx : ArrayLike, optional
            Uncertainty in sea level in meters.
        random_generator : np.random.Generator | int, optional
            Random number generator.

        Returns
        -------
        """
        a0 = -0.133
        a1 = -0.015
        a2 = -1e-4
        a3 = 2.5e-7
        a4 = 1.9e-8
        a5 = 9.6e-11
        y = a0 + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4 + a5 * x**5

        if dx is None:
            return y
        else:
            # Error propagation
            dy = (
                abs(a1 + 2 * a2 * x + 3 * a3 * x**2 + 4 * a4 * x**3 + 5 * a5 * x**4)
                * dx
            )
            rng = np.random.default_rng(self.rng)
            return rng.normal(y, dy)

    # We convert sea level change (SLC) to d18Ow using linear scaling.
    # We assume the following values:
    # d18Owater = +0.0 permil at 0 kyr BP
    # d18Owater = +1.05 permil at -125 m (between 22 and 24 kyr BP in Spratt et al. (2016))
    # This yields the following scaling equation
    def _sealevel_to_d18Ow_linear(self, x, dx):
        # d18Ow = a * z + b
        # a = (d18Ow_lgm) / (z_lgm - z_0)
        # b = - d18Ow_lgm / (z_lgm - z_0)
        # ==> d18Ow = d18Ow_lgm * (z - z_0) / (z_lgm - z_0)
        # d18Ow_lgm = 1.05 +- 0.05
        # z_lgm = -125 +- 5
        # z_0 = z[0]
        d18Ow_lgm = self.rng.normal(1.05, 0.05)
        z_lgm = self.rng.normal(-125, 5)
        z0 = 0
        # z0 = self.rng.normal(x.iloc[0], dx.iloc[0])
        z = self.rng.normal(x, dx)

        return d18Ow_lgm * (z - z0) / (z_lgm - z0)

    def convert(self, x: ArrayLike, method: str = "nonlinear") -> ArrayLike:
        """Convert age to d18Ow.

        Parameters
        ----------
        x : ArrayLike
            Age in calibrated kyr BP.
        method : {"nonlinear", "linear"}
            Conversion method.

        Returns
        -------
        ArrayLike
            d18Ow in permil.

        """
        sealevels = self.sealevel.sel(age=x, method="nearest")

        if method == "nonlinear":
            return self._sealevel_to_d18Ow_nonlinear(
                sealevels["height"], sealevels["stdev"]
            )
        elif method == "linear":
            return self._sealevel_to_d18Ow_linear(
                sealevels["height"], sealevels["stdev"]
            )
        else:
            raise ValueError(
                f"Unknown method: {method}. Choose 'nonlinear' or 'linear'."
            )


# %%
