# %%
import os
import xarray as xr
import pymc as pm
import nutpie
from pymc import Exponential, Normal

from utils.constants import EXPERIMENTS_META, MAP_SPECIES_IDX

# Load data
# =============================================================================
datatree = xr.open_datatree("data/regression/data_prep.nc")


# %%
# Fit models
# =============================================================================
idata = {}
reg_results = {}

for exp_name, exp_meta in EXPERIMENTS_META.items():
    print(f"Experiment: {exp_name}...")

    # Create storage directory
    exp_dir = f"data/regression/{exp_name}"
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # Select experiment data
    data = datatree[exp_meta["data"]].sel(time=exp_meta["time"])
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

    # Fit model
    with pm.Model(coords=coords) as hierarchical_model:
        x = pm.Data(
            "x",
            dataframe["d18Oc"].values,
            dims=("d18Oc"),
        )
        species_idx = pm.Data(
            "species_idx",
            dataframe["species_idx"].values,
            dims=("d18Oc"),
        )

        y = pm.Data("y", dataframe["sigmaT"].values, dims=("d18Oc"))

        # Priors

        # Hyperprior intercept
        prior_intercept_mu = pm.Normal(
            "prior_intercept_mu", dataframe.sigmaT.mean(), sigma=10
        )
        prior_intercept_sigma = pm.Exponential(
            "prior_intercept_sigma", 2.5 * dataframe.sigmaT.std()
        )

        # Hypoerprior slope
        prior_slope_mu = pm.Normal("prior_slope_mu", 0, sigma=10)
        prior_slope_sigma = pm.Exponential(
            "prior_slope_sigma", 2.5 * dataframe.sigmaT.std() / dataframe.d18Oc.std()
        )

        # prior_epsilon_lam = 1 / data_prep.sigmaT.std()
        prior_std = pm.LogNormal("prior_std", mu=0, sigma=1)
        lam = 1 / prior_std**2

        beta0 = Normal(
            "beta0", prior_intercept_mu, sigma=prior_intercept_sigma, dims="species"
        )
        beta1 = Normal("beta1", prior_slope_mu, sigma=prior_slope_sigma, dims="species")
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

    compiled_model = nutpie.compile_pymc_model(hierarchical_model)
    idata[exp_name] = nutpie.sample(compiled_model)

    with hierarchical_model:
        print("Prior predictive...", flush=True)
        prior_predictive_linear_hr = pm.sample_prior_predictive(
            compile_kwargs={"mode": "NUMBA"}
        )
        print("Posterior predictive...", flush=True)
        pm.sample_posterior_predictive(
            idata[exp_name],
            var_names=["sigmaT"],
            compile_kwargs={"mode": "NUMBA"},
            extend_inferencedata=True,
        )
        print("Log likelihood...", flush=True)
        pm.compute_log_likelihood(idata[exp_name])

    idata[exp_name].extend(prior_predictive_linear_hr)

    # Mean of posterior predictive
    y_model = idata[exp_name].posterior_predictive.sigmaT.mean(["chain", "draw"])
    dataframe["y_hat"] = y_model
    dataframe["residuals"] = y_model - dataframe["sigmaT"]
    reg_results[exp_name] = dataframe

    # Save results
    idata[exp_name].to_netcdf(f"{exp_dir}/idata.nc")
    dataframe.to_csv(f"{exp_dir}/regression_results.csv")

# %%
