# %%
import arviz as az
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import nutpie
import pandas as pd
import preliz as pz
import pymc as pm
import os
import regionmask as rm
import xarray as xr
import xarray_regrid
from cartopy.feature import LAND
from pymc import Exponential, Gamma, Normal

from utils.constants import MAP_SPECIES_IDX

# %%
# Load data
# =============================================================================

data = xr.open_datatree("data/regression/data_prep.nc")

# Select experiment data
data = data.to_dataset()
dataframe = (
    data.stack(sample=("lon", "lat", "species"))
    .dropna("sample")
    .to_dataframe()
    .reset_index(drop=True)
)

print("Number of samples: ", dataframe.shape[0])

# Create coordinates
dataframe["species_idx"] = dataframe["species"].map(MAP_SPECIES_IDX)

coords = {
    "sample": dataframe.index.values,
    "d18Oc": dataframe["d18Oc"].values,
    "sigmaT": dataframe["sigmaT"].values,
    "feature": {"lon": dataframe["lon"].values, "lat": dataframe["lat"].values},
    "lon": dataframe.lon,
    "lat": dataframe.lat,
    "species_id": list(MAP_SPECIES_IDX.values()),
    "species": list(MAP_SPECIES_IDX.keys()),
    "species_idx": dataframe["species_idx"].values,
}


# Meta information
# =============================================================================
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

all_idata = {}
all_reg_results = {}


# Prepare modelling
# =============================================================================
EXPERIMENTS = ["poly1_pool", "poly2_pool", "poly1_hier"]

# Create storage directory
for exp in EXPERIMENTS:
    exp_dir = f"data/regression/obs/{exp}"
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)


# %%
# Fit models
# =============================================================================
def fit_model(dataframe, coords, model):
    dataframe = dataframe.copy(deep=True)

    match model:
        case "poly1_pool":
            with pm.Model(coords=coords) as model:
                X = pm.Data(
                    "x",
                    dataframe["d18Oc"].values,
                    dims=("d18Oc"),
                )
                y = pm.Data("y", dataframe["sigmaT"].values, dims=("d18Oc"))

                # Priors
                prior_intercept_mu = dataframe.sigmaT.mean()
                prior_intercept_sigma = 2.5 * dataframe.sigmaT.std()
                prior_slope_mu = 0
                prior_slope_sigma = 2.5 * dataframe.sigmaT.std() / dataframe.d18Oc.std()
                prior_epsilon_lam = 1 / dataframe.sigmaT.std()
                beta0 = Normal("beta0", prior_intercept_mu, sigma=prior_intercept_sigma)
                beta1 = Normal("beta1", prior_slope_mu, sigma=prior_slope_sigma)
                sigma = Exponential("sigma", lam=prior_epsilon_lam)

                # Define likelihood
                mu = pm.Deterministic("mu", beta0 + beta1 * X, dims=("d18Oc"))
                likelihood = Normal(
                    "sigmaT",
                    mu=mu,
                    sigma=sigma,
                    observed=y,
                    dims=("d18Oc"),
                )

        case "poly2_pool":
            with pm.Model(coords=coords) as model:
                x1 = pm.Data(
                    "x1",
                    dataframe["d18Oc"].values,
                    dims=("d18Oc"),
                )
                x2 = pm.Data(
                    "x2",
                    dataframe["d18Oc"].values ** 2,
                    dims=("d18Oc"),
                )
                y = pm.Data("y", dataframe["sigmaT"].values, dims=("d18Oc"))

                # Priors
                prior_epsilon_lam = 1 / dataframe.sigmaT.std()
                beta = pz.maxent(pz.Normal(), lower=-10, upper=10, mass=0.9, plot=False)
                beta0 = beta.to_pymc("beta0")
                beta1 = beta.to_pymc("beta1")
                beta2 = beta.to_pymc("beta2")
                sigma = Exponential("sigma", lam=prior_epsilon_lam)

                # Define likelihood
                mu = pm.Deterministic(
                    "mu", beta0 + beta1 * x1 + beta2 * x2, dims=("d18Oc")
                )
                likelihood = Normal(
                    "sigmaT",
                    mu=mu,
                    sigma=sigma,
                    observed=y,
                    dims=("d18Oc"),
                )

        case "poly1_hier":
            with pm.Model(coords=coords) as model:
                x = pm.Data(
                    "x",
                    dataframe["d18Oc"].values,  # shape (n,)
                    dims=("d18Oc"),
                )
                species_idx = pm.Data(
                    "species_idx",
                    dataframe["species_idx"].values,
                    dims=("d18Oc"),
                )

                y = pm.Data(
                    "y", dataframe["sigmaT"].values, dims=("d18Oc")
                )  # shape (n,)

                # Priors

                # Hyperprior intercept
                prior_intercept_mu = pm.Normal(
                    "prior_beta0_mu", dataframe.sigmaT.mean(), sigma=10
                )
                prior_intercept_sigma = pm.Exponential(
                    "prior_beta0_sigma", 2.5 * dataframe.sigmaT.std()
                )

                # Hypoerprior slope
                prior_slope_mu = pm.Normal("prior_beta1_mu", 0, sigma=10)
                prior_slope_sigma = pm.Exponential(
                    "prior_beta1_sigma",
                    2.5 * dataframe.sigmaT.std() / dataframe.d18Oc.std(),
                )

                # prior_epsilon_lam = 1 / dataframe.sigmaT.std()
                prior_std = pm.LogNormal("prior_std", mu=0, sigma=1)
                lam = 1 / prior_std**2

                beta0 = Normal(
                    "beta0",
                    prior_intercept_mu,
                    sigma=prior_intercept_sigma,
                    dims="species",
                )
                beta1 = Normal(
                    "beta1", prior_slope_mu, sigma=prior_slope_sigma, dims="species"
                )
                sigma = Exponential("sigma", lam=lam, dims="species")

                # Define likelihood
                mu = pm.Deterministic(
                    "mu", beta0[species_idx] + beta1[species_idx] * x, dims=("d18Oc")
                )
                likelihood = Normal(
                    "sigmaT",
                    mu=mu,
                    sigma=sigma[species_idx],
                    observed=y,
                    dims=("d18Oc"),
                )

        case _:
            raise ValueError("Invalid model")

    compiled_model = nutpie.compile_pymc_model(model)
    idata = nutpie.sample(compiled_model)

    with model:
        print("Prior predictive...", flush=True)
        prior_predictive = pm.sample_prior_predictive(compile_kwargs={"mode": "NUMBA"})
        print("Posterior predictive...", flush=True)
        pm.sample_posterior_predictive(
            idata,
            var_names=["sigmaT"],
            compile_kwargs={"mode": "NUMBA"},
            extend_inferencedata=True,
        )
        print("Log likelihood...", flush=True)
        pm.compute_log_likelihood(idata)

    idata.extend(prior_predictive)

    # Mean of posterior predictive
    y_model = idata.posterior_predictive.sigmaT.mean(["chain", "draw"])
    dataframe["y_hat"] = y_model
    dataframe["residuals"] = y_model - dataframe["sigmaT"]

    return idata, dataframe


for exp in EXPERIMENTS:
    print(f"Experiment: {exp}...")

    idata, reg_result = fit_model(dataframe, coords, exp)

    idata.to_netcdf(f"data/regression/obs/{exp}/idata.nc")
    reg_result.to_csv(f"data/regression/obs/{exp}/regression_results.csv")

    all_idata[exp] = idata
    all_reg_results[exp] = reg_result
