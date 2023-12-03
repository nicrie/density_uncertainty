import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances

from pymc_models import MODELS
from utils.constants import AVG_EARTH_RADIUS
from utils.kernels import bisquare_kernel
from utils.linalg import ones_vector


def load_model(name: str):
    """Load a model.

    Parameters
    ----------
    name : str
        Name of the model to load. Options are: 'Linear', 'Poly2', 'LinearESF', 'Poly2ESF'.

    Returns
    -------
    pymc.Model
        Trained model.

    """
    model = MODELS[name].load(f"data/regression/full/{name}/model.nc")
    return model


def nystroem_approximation(latlons: pd.DataFrame):
    """Compute the approximate eigenvectors using the Nystroem approximation.

    Parameters
    ----------
    latlons : pd.DataFrame
        DataFrame containing the latitude and longitude of the samples.

    """
    # Cluster centroids
    centroids = pd.read_csv("data/esf/full/centroids.csv").values
    n_clusters = centroids.shape[0]
    rads_centroids = np.deg2rad(centroids)
    # Regularized kernel matrix of centroids
    CLplus = pd.read_csv("data/esf/full/CLplus.csv").values
    # Eigenvectors of the regularized kernel matrix
    EL = pd.read_csv("data/esf/full/EL.csv")  # Eigenvectors of the Laplacian
    col_names = EL.columns
    EL = EL.values
    # eigenvalues of the regularized kernel matrix
    eigvalsL = pd.read_csv("data/esf/full/eigvalsL.csv").values.squeeze()

    # Coordinates of new samples
    n_samples = latlons.shape[0]
    new_rads = np.deg2rad(latlons)
    new_dist = haversine_distances(new_rads, rads_centroids) * AVG_EARTH_RADIUS
    # Bandwidth used in bisquare kernel (in km)
    bandwidth = 16637
    C2L = bisquare_kernel(new_dist, bandwidth)
    E_new = (
        (
            C2L
            - np.kron(
                ones_vector(n_samples),
                ones_vector(n_clusters).T @ CLplus / n_clusters,
            )
        )
        @ EL
        @ np.diag(1.0 / (eigvalsL))
    )
    return pd.DataFrame(E_new, columns=col_names)


def predict_sigmaT(
    X: pd.DataFrame,
    model: str = "Poly2ESF",
    name_d18Oc: str = "d18Oc",
    name_d18Oc_stdev: str = "d18Oc_stdev",
    name_lon: str = "lon",
    name_lat: str = "lat",
    random_seed=None,
):
    """Predict sigmaT for a given dataset.

    Parameters
    ----------
    X : xr.DataArray
        Predictors used to predict sigmaT.
    model : str, optional
        Model to use for prediction. Options are:
        - "Linear": Normal linear regression.
        - "Poly2": Second order polynomial regression.
        - "LinearESF": Linear regression with Eigenvector Spatial Filtering (ESF).
        - "Poly2ESF": Second order polynomial regression with ESF. (default)

    Returns
    -------
    xr.DataArray
        Dataset with predicted sigmaT.

    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if model not in ["Linear", "Poly2", "LinearESF", "Poly2ESF"]:
        raise ValueError(
            "model must be one of 'Linear', 'Poly2', 'LinearESF', 'Poly2ESF'."
        )
    if name_d18Oc not in X.columns:
        raise ValueError(f"{name_d18Oc} is not present in X.")
    if name_d18Oc_stdev not in X.columns:
        raise ValueError(f"{name_d18Oc_stdev} is not present in X.")

    if model in ["LinearESF", "Poly2ESF"]:
        if name_lat not in X.columns:
            raise ValueError(
                f"Fitting an ESF model requires {name_lat} to be present in X."
            )
        if name_lon not in X.columns:
            raise ValueError(
                f"Fitting an ESF model requires {name_lon} to be present in X."
            )

    X = X.reset_index(inplace=False, drop=True)
    predictors = X[[name_d18Oc, name_d18Oc_stdev]]

    if model in ["LinearESF", "Poly2ESF"]:
        # For spatial models only
        latlons = X[[name_lat, name_lon]]
        E_new = nystroem_approximation(latlons)
        predictors = pd.concat([predictors, E_new], axis=1)

    loaded_model = load_model(model)
    pred = loaded_model.predict_posterior(predictors, random_seed=random_seed)
    pred.coords.update({"d18Oc": X[name_d18Oc].values})
    return pred
