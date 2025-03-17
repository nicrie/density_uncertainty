import numpy as np
import pymc as pm
import pytensor.tensor as pt
import numbers
import numpy as np
from pytensor.tensor.nlinalg import matrix_inverse, det
from pytensor import scan
from pymc.gp.cov import Covariance

from pytensor.tensor.variable import TensorVariable

TensorLike = np.ndarray | TensorVariable


class Matern32Haversine(pm.gp.cov.Stationary):
    def __init__(self, input_dims, ls, r=6378.137, active_dims=None):
        if input_dims != 2:
            raise ValueError("Great circle distance is only defined on 2 dimensions")
        super().__init__(input_dims, ls=ls, active_dims=active_dims)
        self.r = r

    def great_circle_distance(self, X, Xs=None):
        if Xs is None:
            Xs = X

        # Assume first column is longitude and second is latitude
        lat1_ = pt.deg2rad(X[:, 1])
        lon1_ = pt.deg2rad(X[:, 0])
        lat2_ = pt.deg2rad(Xs[:, 1])
        lon2_ = pt.deg2rad(Xs[:, 0])

        # Reshape lon/lat into 2D
        lat1 = lat1_[:, None]
        lon1 = lon1_[:, None]

        # Elementwise differnce of lats and lons
        dlat = lat2_ - lat1
        dlon = lon2_ - lon1

        # Compute haversine
        d = pt.sin(dlat / 2) ** 2 + pt.cos(lat1) * pt.cos(lat2_) * pt.sin(dlon / 2) ** 2
        return self.r * 2 * pt.arcsin(pt.sqrt(d)) + 1e-12

    def full(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        r = self.great_circle_distance(X, Xs)
        return (1.0 + np.sqrt(3.0) * pt.mul(r, 1.0 / self.ls)) * pt.exp(
            -np.sqrt(3.0) * pt.mul(r, 1.0 / self.ls)
        )


class Matern32Chordal(pm.gp.cov.Stationary):
    def __init__(self, input_dims, ls, r=6378.137, active_dims=None):
        if input_dims != 2:
            raise ValueError("Chordal distance is only defined on 2 dimensions")
        super().__init__(input_dims, ls=ls, active_dims=active_dims)
        self.r = r

    def lonlat2xyz(self, lonlat):
        lonlat = pt.deg2rad(lonlat)
        return self.r * pt.stack(
            [
                pt.cos(lonlat[..., 0]) * pt.cos(lonlat[..., 1]),
                pt.sin(lonlat[..., 0]) * pt.cos(lonlat[..., 1]),
                pt.sin(lonlat[..., 1]),
            ],
            axis=-1,
        )

    def chordal_dist(self, X, Xs=None):
        if Xs is None:
            Xs = X
        X, Xs = pt.broadcast_arrays(
            self.lonlat2xyz(X[..., :, None, :]), self.lonlat2xyz(Xs[..., None, :, :])
        )
        return pt.sqrt(pt.sum(((X - Xs) / self.ls) ** 2, axis=-1) + 1e-12)

    def full(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        r = self.chordal_dist(X, Xs)
        return (1.0 + np.sqrt(3.0) * r) * pt.exp(-np.sqrt(3.0) * r)


# Locally anisotropic nonstationry covariance function (Cao 2024)
# =============================================================================

# ------------------------------------------------------------------------------
# Helper functions (all using PyTensor operations)
# ------------------------------------------------------------------------------


def spherical_to_cartesian(lon, lat):
    """
    Convert spherical coordinates (lon, lat) in radians to Cartesian (x, y, z).
    Assumes lon and lat are PyTensor variables (can be vectors).
    """
    x = pt.cos(lat) * pt.cos(lon)
    y = pt.cos(lat) * pt.sin(lon)
    z = pt.sin(lat)
    return pt.stack([x, y, z], axis=-1)


def Rx(theta):
    """Rotation matrix about x-axis."""
    return pt.stack(
        [
            pt.stack(
                [
                    pt.as_tensor_variable(1.0),
                    pt.as_tensor_variable(0.0),
                    pt.as_tensor_variable(0.0),
                ]
            ),
            pt.stack([pt.as_tensor_variable(0.0), pt.cos(theta), -pt.sin(theta)]),
            pt.stack([pt.as_tensor_variable(0.0), pt.sin(theta), pt.cos(theta)]),
        ],
        axis=0,
    )


def Ry(theta):
    """Rotation matrix about y-axis."""
    return pt.stack(
        [
            pt.stack([pt.cos(theta), pt.as_tensor_variable(0.0), pt.sin(theta)]),
            pt.stack(
                [
                    pt.as_tensor_variable(0.0),
                    pt.as_tensor_variable(1.0),
                    pt.as_tensor_variable(0.0),
                ]
            ),
            pt.stack([-pt.sin(theta), pt.as_tensor_variable(0.0), pt.cos(theta)]),
        ],
        axis=0,
    )


def Rz(theta):
    """Rotation matrix about z-axis."""
    return pt.stack(
        [
            pt.stack([pt.cos(theta), -pt.sin(theta), pt.as_tensor_variable(0.0)]),
            pt.stack([pt.sin(theta), pt.cos(theta), pt.as_tensor_variable(0.0)]),
            pt.stack(
                [
                    pt.as_tensor_variable(0.0),
                    pt.as_tensor_variable(0.0),
                    pt.as_tensor_variable(1.0),
                ]
            ),
        ],
        axis=0,
    )


def local_anisotropy_matrix(lon, lat, beta1, beta2, kappa):
    """
    For a single location (lon, lat), compute the 3x3 local anisotropy matrix Sigma(s).
    beta1 and beta2 are 3-element vectors and kappa a scalar.
    """
    # Compute local scaling parameters
    g1 = pt.exp(beta1[0] + beta1[1] * pt.sin(lon) + beta1[2] * lat)
    g2 = pt.exp(beta2[0] + beta2[1] * pt.sin(lon) + beta2[2] * lat)
    D = pt.diag(pt.stack([pt.as_tensor_variable(1.0), g1, g2]))
    # Local rotation in the tangent plane
    R_x = Rx(kappa)
    Sigma_tilde = pt.dot(R_x, pt.dot(D, R_x.T))
    # Rotate to global coordinates via the spherical rotation
    R_global = pt.dot(Rz(lon), Ry(lat))
    Sigma = pt.dot(R_global, pt.dot(Sigma_tilde, R_global.T))
    return Sigma


def matern_cov(q, nu, rng):
    """
    Compute the Matérn correlation for a given distance q.
    Uses the formula
      ρ(q) = (2^(1-ν)/Γ(ν)) (√(2ν)*q/rng)^ν K_ν(√(2ν)*q/rng)
    where K_ν is the modified Bessel function of the second kind.
    For q==0 the limit is 1.
    """
    arg = pt.sqrt(2 * nu) * q / rng
    factor = 2 ** (1 - nu) / pt.gamma(nu)
    # Use a small epsilon to avoid zero division.
    matern_val = factor * pt.power(arg, nu) * pt.besselK(arg, nu)
    matern_val = pt.switch(pt.eq(q, 0), pt.as_tensor_variable(1.0), matern_val)
    return matern_val


# We vectorize the computation of Sigma(s) for a set of points using a scan.
def compute_Sigma_vec(lon_vec, lat_vec, beta1, beta2, kappa):
    """
    Given vectors of longitudes and latitudes (each shape (n,)),
    return a tensor of shape (n, 3, 3) where each slice is Sigma(s) at that point.
    """

    def body(lon, lat):
        return local_anisotropy_matrix(lon, lat, beta1, beta2, kappa)

    Sigma_seq, _ = scan(
        fn=lambda lon, lat: (body(lon, lat),), sequences=[lon_vec, lat_vec]
    )
    return Sigma_seq


def compute_cov_row(i, cartX, Sigma_X, cartXs, Sigma_Xs, nu, rng):
    """
    Compute the covariance between a fixed point i and all points in another set.
    cartX: (n, 3), Sigma_X: (n, 3, 3)
    cartXs: (m, 3), Sigma_Xs: (m, 3, 3)
    Returns a vector of shape (m,).
    """
    # diff: shape (m, 3)
    diff = cartX[i] - cartXs
    # For fixed i, add Sigma_X[i] to every Sigma_Xs[j]:
    Sigma_sum = Sigma_X[i][None, :, :] + Sigma_Xs  # shape (m, 3, 3)
    invSigma_sum = matrix_inverse(Sigma_sum)  # shape (m, 3, 3)
    # Compute squared Mahalanobis distance: q^2 = 2 * diff^T invSigma_sum diff
    q2 = 2 * pt.einsum("ij,ijk,ik->i", diff, invSigma_sum, diff)
    q = pt.sqrt(pt.clip(q2, 1e-12, np.inf))
    # Determinants for normalization:
    det_i = det(Sigma_X[i])
    det_Xs = det(Sigma_Xs)  # shape (m,)
    det_avg = det(Sigma_sum / 2)  # shape (m,)
    c_val = (pt.power(det_i, 0.25) * pt.power(det_Xs, 0.25)) / pt.power(det_avg, 0.5)
    return c_val * matern_cov(q, nu, rng)


# ------------------------------------------------------------------------------
# Custom Covariance Class
# ------------------------------------------------------------------------------


class AnisotropicSphere(Covariance):
    """
    Nonstationary, locally anisotropic covariance function on the sphere.

    Each input is a 2D vector [lon, lat] (in radians). The kernel is defined as
    K(s_i,s_j) = c(s_i,s_j) * ρ( q(s_i,s_j) ),

    where q(s_i,s_j) and c(s_i,s_j) are computed from the local anisotropy matrices:

        Σ(s) = Rz(lon) Ry(lat) [ Rx(κ) diag(1,γ₁(s),γ₂(s)) Rx(κ)ᵀ ] Ry(lat)ᵀ Rz(lon)ᵀ,

    with
        γ₁(s) = exp(β₁₀ + β₁₁ sin(lon) + β₁₂ lat),
        γ₂(s) = exp(β₂₀ + β₂₁ sin(lon) + β₂₂ lat).

    The base correlation ρ is given by a Matérn covariance with smoothness ν and range parameter.
    """

    def __init__(self, input_dim, beta1, beta2, kappa, nu, ls, active_dims=None):
        """
        Parameters
        ----------
        input_dim : int
            Should be 2 (for lon,lat).
        beta1 : array-like of shape (3,)
            Coefficients for computing γ₁(s).
        beta2 : array-like of shape (3,)
            Coefficients for computing γ₂(s).
        kappa : scalar
            Local rotation parameter (in radians).
        nu : scalar
            Smoothness parameter for the Matérn correlation.
        ls : scalar
            Range parameter for the Matérn correlation.
        active_dims : sequence of ints, optional
            Which dimensions of the input to use (default: all).
        """
        super().__init__(input_dim, active_dims)
        self.beta1 = pt.as_tensor_variable(beta1)
        self.beta2 = pt.as_tensor_variable(beta2)
        self.kappa = pt.as_tensor_variable(kappa)
        self.nu = pt.as_tensor_variable(nu)
        self.ls = pt.as_tensor_variable(ls)

    def diag(self, X):
        # By construction, K(s,s)=1.
        X, _ = self._slice(X, None)
        return pt.alloc(1.0, X.shape[0])

    def full(self, X, Xs=None):
        # Slice inputs (X assumed to be shape (n,2))
        X, Xs = self._slice(X, Xs)
        if Xs is None:
            Xs = X
        # Compute Cartesian coordinates (each of shape (n,3) or (m,3))
        cartX = spherical_to_cartesian(X[:, 0], X[:, 1])
        cartXs = spherical_to_cartesian(Xs[:, 0], Xs[:, 1])
        # Compute local anisotropy matrices for each point (shape (n,3,3) and (m,3,3))
        Sigma_X = compute_Sigma_vec(
            X[:, 0], X[:, 1], self.beta1, self.beta2, self.kappa
        )
        Sigma_Xs = compute_Sigma_vec(
            Xs[:, 0], Xs[:, 1], self.beta1, self.beta2, self.kappa
        )

        # Define a function that computes one row of the covariance matrix.
        def row_fn(i, cartX, Sigma_X, cartXs, Sigma_Xs, nu, ls):
            return compute_cov_row(i, cartX, Sigma_X, cartXs, Sigma_Xs, nu, ls)

        # Use pt.arange and pt.scan to compute each row.
        n = X.shape[0]
        cov_rows, updates = scan(
            fn=lambda i, cartX, Sigma_X, cartXs, Sigma_Xs, nu, rng: row_fn(
                i, cartX, Sigma_X, cartXs, Sigma_Xs, nu, rng
            ),
            sequences=pt.arange(n),
            non_sequences=[cartX, Sigma_X, cartXs, Sigma_Xs, self.nu, self.ls],
        )

        return cov_rows
