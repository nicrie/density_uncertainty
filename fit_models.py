# %%
# import multiprocessing as mp

import os
import warnings
from importlib import reload
from pprint import pprint

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
import xarray as xr
from esda.moran import Moran
from libpysal.weights import full2W

import pymc_models

# mp.set_start_method("forkserver")
warnings.simplefilter("ignore")


reload(pymc_models)
# %%


def gamma(zMC, npos):
    alpha = 0.1742
    numerator = 2.1480 - 6.1808 * (zMC + 0.6) ** alpha
    denominator = npos ** (0.1298) + 3.3534 / (zMC + 0.6) ** alpha
    return np.exp(numerator / denominator)


def load_files(path):
    data = {}
    # get all files in data folder
    for file in os.listdir(path):
        # check if file is a csv file
        if file.endswith(".csv"):
            # load csv file into a pandas dataframe
            data[file[:-4]] = pd.read_csv(os.path.join(path, file))
    return data


def get_sampler_config(model_name):
    if model_name in non_spatial_models:
        return sampler_configs["non_spatial"]
    elif model_name in spatial_models:
        return sampler_configs["spatial"]
    else:
        raise ValueError("Model type not recognized.")


def get_rhs_config(exp: str, model_name: str, X: pd.DataFrame):
    if model_name not in spatial_models:
        return {}
    # RHS
    n = X.shape[0]
    npos = X.loc[:, "E0":].shape[1]
    D = npos
    zMC = zMoransCoef[exp][model_name]
    p0 = npos / (1 + gamma(zMC, npos))
    data_sigma = stdev_data_resid[exp][model_name]  # precomputed
    tau0 = p0 / (D - p0) / np.sqrt(n)
    s_tau = tau0 * data_sigma
    sc = 0.5 * data_sigma / X.loc[:, "E0":].std(0)[0]
    vc = 15
    c2_alpha = vc / 2
    c2_beta = vc * sc**2 / 2
    tau_alpha = s_tau
    print("Experiment: {:} | {:}".format(exp, model_name))
    print("p0: {:.02f}".format(p0))
    print("tau0: {:.02f}".format(tau0))
    print("s_tau: {:.02f}".format(s_tau))
    print("sc: {:.02f}".format(sc))
    print("c2_alpha: {:.02f}".format(c2_alpha))
    print("c2_beta: {:.02f}".format(c2_beta))

    return {
        "n_eigenvectors": npos,
        "c2_alpha": c2_alpha,
        "c2_beta": c2_beta,
        "tau_alpha": tau_alpha,
    }


if __name__ == "__main__":
    # %%
    # Load data
    # =============================================================================
    print("Loading data...")

    EXPERIMENTS = ["full", "ruber"]

    files = {}
    for exp in EXPERIMENTS:
        files[exp] = load_files(f"data/esf/{exp}")

    # %%
    # Prepare data
    # =============================================================================
    print("Preparing data...")

    data_ext = {}
    Xdata = {}
    ydata = {}
    for exp in EXPERIMENTS:
        ev_columns = files[exp]["E_approx"].columns

        dext = pd.concat(
            [files[exp]["database_clustered"], files[exp]["E_approx"]], axis=1
        )
        # We assume a measurement error of 0.1‰ for the d18Oc measurements
        dext["d18Oc_stdev"] = 0.1

        data_ext[exp] = dext
        Xdata[exp] = dext[["d18Oc", "d18Oc_stdev"] + list(ev_columns)]
        ydata[exp] = dext["sigmaT"]

    # Save extended dataset
    for exp in EXPERIMENTS:
        path = f"data/regression/{exp}/"
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, "database_extended.csv")
        data_ext[exp].to_csv(file_path, index=False)

    # %%
    # Model configuration
    # =============================================================================
    print("Configuring models...")

    non_spatial_models = ["Linear", "Poly2"]
    spatial_models = ["LinearESF", "Poly2ESF"]

    model_types = {
        # "Linear": pymc_models.Linear,
        # "Poly2": pymc_models.Poly2,
        "LinearESF": pymc_models.LinearRHSESF,
        "Poly2ESF": pymc_models.Poly2RHSESF,
    }

    bandwidth = {"full": 16637, "ruber": 16391}

    # z value of Moran's I (precomputed)
    zMoransCoef = {
        "full": {"LinearESF": 479.0, "Poly2ESF": 193.0},
        "ruber": {"LinearESF": 45.0, "Poly2ESF": 27.0},
    }

    # standard deviation of sigmaT residuals (precomputed)
    stdev_data_resid = {
        "full": {"LinearESF": 1.0, "Poly2ESF": 0.8},
        "ruber": {"LinearESF": 0.9, "Poly2ESF": 0.8},
    }

    sampler_configs = {
        "non_spatial": {
            "draws": 2_000,
            "tune": 50_000,
            "chains": 4,
            "target_accept": 0.5,
            "cores": 4,
        },
        "spatial": {
            "draws": 2_000,
            "tune": 150_000,
            "chains": 4,
            "target_accept": 0.9,
            "cores": 2,
            "mp_ctx": "forkserver",
        },
    }

    NONSPATIAL_MODEL_CONFIG: dict = {
        "a0_mu_prior": 0.0,
        "a1_mu_prior": 0.0,
        "a0_sigma_prior": 1000.0,
        "a1_sigma_prior": 1000.0,
        "sigma_y_alpha_prior": 1.0,
        "sigma_y_beta_prior": 1.0,
    }
    SPATIAL_MODEL_CONFIG: dict = NONSPATIAL_MODEL_CONFIG | {
        "n_eigenvectors": 1.0,
        "c2_alpha": 1.0,
        "c2_beta": 1.0,
        "tau_alpha": 1.0,
        "lbda_alpha": 1.0,
        "b_mu_prior": 0.0,
    }
    model_configs = {
        "Linear": NONSPATIAL_MODEL_CONFIG.copy(),
        "Poly2": NONSPATIAL_MODEL_CONFIG
        | {"a2_mu_prior": 0.0, "a2_sigma_prior": 1000.0},
        "LinearESF": SPATIAL_MODEL_CONFIG.copy(),
        "Poly2ESF": SPATIAL_MODEL_CONFIG
        | {"a2_mu_prior": 0.0, "a2_sigma_prior": 1000.0},
    }

    def get_model_config(exp: str, model_name: str, X: pd.DataFrame = None, **kwargs):
        model_config = model_configs[model_name].copy()

        if model_name in spatial_models:
            rhs_config = get_rhs_config(exp, model_name, X)
            model_config.update(rhs_config)

        return model_config

    models = {}
    idata = {}

    # %%
    # Run models
    # =============================================================================
    print("Fit models...")
    for exp in EXPERIMENTS:
        models[exp] = {}
        idata[exp] = {}
        for name, model in model_types.items():
            print("\n\nExperiment: {:} | {:}".format(exp, name))
            print("-" * 80)
            print()
            X = Xdata[exp]
            y = ydata[exp]

            sampler_config = get_sampler_config(name)
            model_config = get_model_config(exp, name, X)

            # Fit model
            m = model(model_config=model_config, sampler_config=sampler_config)
            infdata = m.fit(X, y)

            # Compute log likelihood
            print("Computing log likelihood...")
            try:
                with m.model:
                    pm.compute_log_likelihood(infdata)
            except ValueError:
                pass

            # Save model
            print("Saving model...")
            path = f"data/regression/{exp}/{name}/"
            os.makedirs(path, exist_ok=True)
            file_model = os.path.join(path, "model.nc")
            file_inference_data = os.path.join(path, "inference_data.nc")
            m.save(file_model)
            az.to_netcdf(infdata, file_inference_data)

            models[exp][name] = m
            idata[exp][name] = infdata
