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
    exp_dir = f"data/regression/sim/{exp_name}"
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # Select experiment data
    data_train = datatree[exp_meta["data"]].sel(time="PI")
    data_test = datatree[exp_meta["data"]].sel(time="LGM")

    data_train = data_train.to_dataset()
    data_test = data_test.to_dataset()

    dataframe_train = (
        data_train.stack(sample=("lon", "lat", "species"))
        .dropna("sample")
        .to_dataframe()
        .reset_index(drop=True)
    )
    dataframe_test = (
        data_test.stack(sample=("lon", "lat", "species"))
        .dropna("sample")
        .to_dataframe()
        .reset_index(drop=True)
    )

    print("Number of training samples: ", dataframe_train.shape[0])
    print("Number of test samples: ", dataframe_test.shape[0])

    # Create coordinates
    dataframe_train["species_idx"] = dataframe_train["species"].map(MAP_SPECIES_IDX)
    dataframe_test["species_idx"] = dataframe_test["species"].map(MAP_SPECIES_IDX)

    coords = {
        "d18Oc": dataframe_train["d18Oc"].values,
        "d18Oc_pred": dataframe_test["d18Oc"].values,
        "sigmaT": dataframe_train["sigmaT"].values,
        "feature": {
            "lon": dataframe_train["lon"].values,
            "lat": dataframe_train["lat"].values,
        },
        "lon": dataframe_train.lon,
        "lat": dataframe_train.lat,
        "species_id": list(MAP_SPECIES_IDX.values()),
        "species": list(MAP_SPECIES_IDX.keys()),
        "species_idx": dataframe_train["species_idx"].values,
        "species_idx_pred": dataframe_test["species_idx"].values,
    }

    # Fit model
    with pm.Model(coords=coords) as hierarchical_model:
        x = pm.Data(
            "x",
            dataframe_train["d18Oc"].values,
            dims=("d18Oc"),
        )
        species_idx = pm.Data(
            "species_idx",
            dataframe_train["species_idx"].values,
            dims=("d18Oc"),
        )

        y = pm.Data("y", dataframe_train["sigmaT"].values, dims=("d18Oc"))

        # Priors

        # Hyperprior intercept
        prior_intercept_mu = pm.Normal(
            "prior_intercept_mu", dataframe_train.sigmaT.mean(), sigma=10
        )
        prior_intercept_sigma = pm.Exponential(
            "prior_intercept_sigma", 2.5 * dataframe_train.sigmaT.std()
        )

        # Hypoerprior slope
        prior_slope_mu = pm.Normal("prior_slope_mu", 0, sigma=10)
        prior_slope_sigma = pm.Exponential(
            "prior_slope_sigma",
            2.5 * dataframe_train.sigmaT.std() / dataframe_train.d18Oc.std(),
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

    # Mean of posterior predictive (train; PI)
    y_model = idata[exp_name].posterior_predictive.sigmaT.mean(["chain", "draw"])
    dataframe_train["y_hat"] = y_model
    dataframe_train["residuals"] = y_model - dataframe_train["sigmaT"]
    reg_results[exp_name] = dataframe_train

    # Make prediction on test data (test; LGM)
    beta0 = idata[exp_name].posterior["beta0"]
    beta1 = idata[exp_name].posterior["beta1"]

    x_new = (
        dataframe_test.reset_index()
        .set_index(["index", "species"])["d18Oc"]
        .to_xarray()
    )
    x_new = x_new.rename({"index": "d18Oc"})
    x_new = x_new.assign_coords({"d18Oc": dataframe_test["d18Oc"]})
    pred = beta0 + beta1 * x_new
    pred = pred.max("species")
    pred = pred.rename({"d18Oc": "d18Oc_pred"})
    pred.name = "sigmaT_pred"
    idata[exp_name].add_groups({"prediction": pred})

    # Mean of posterior predictive (test; LGM)
    y_model = idata[exp_name].prediction.sigmaT_pred.mean(["chain", "draw"])
    dataframe_test["y_hat"] = y_model
    dataframe_test["residuals"] = y_model - dataframe_test["sigmaT"]

    # Save results
    idata[exp_name].to_netcdf(f"{exp_dir}/idata.nc")
    dataframe_train.to_csv(f"{exp_dir}/regression_results.csv")
    dataframe_test.to_csv(f"{exp_dir}/regression_results_test.csv")

# %%
