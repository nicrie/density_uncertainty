def salinity_approx(rho, T, P):
    """Calculate salinity based on Munhoven's approximation.

    Parameters
    ----------
    rho : float
        Density as sigma-t; defined as rho(S, T) - 1000 kg m$^{-3}$.
    T : float
        Temperature in K.
    P : float
        Pressure in bar.

    Returns
    -------
    float
        Salinity in psu.

    """
    a0 = 1040.0145
    a1 = 0.77629393
    a2 = -0.25013591
    a3 = 4.206266e-2
    a4 = -4.7473116e-3
    a5 = -4.7974224e-6
    a6 = -2.140492e-4

    T0 = 285.65  # in K
    P0 = 300  # in bar
    S0 = 35.5

    return (
        1
        / a1
        * (
            rho
            - a0
            - a2 * (T - T0)
            - a3 * (P - P0)
            - a4 * (T - T0) ** 2
            - a5 * (P - P0) ** 2
            - a6 * (T - T0) * (P - P0)
        )
        + S0
    )
