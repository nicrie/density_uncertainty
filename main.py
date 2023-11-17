# %%
import os

import pandas as pd
import numpy as np
import xarray as xr
import utils.visualization as vis
from importlib import reload

from models import predict_sigmaT
from utils.datasets import load_dataset


def draw_samples(mu, sigma, n_draws, dim=1000, seed=None):
    np.random.seed(seed)
    samples = xr.apply_ufunc(
        np.random.normal,
        mu,
        sigma,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[["draw", dim]],
        kwargs=dict(size=(n_draws, mu[dim].size)),
    )
    samples.coords.update({"draw": samples.coords["draw"]})
    return samples


def load_input_data(path, **kwargs):
    # get the file extension of path
    file_extension = os.path.splitext(path)[1]
    VALID_EXTENSIONS = [".csv", ".txt", ".xlsx", ".xls"]
    if file_extension in [".csv", ".txt"]:
        return pd.read_csv(path, **kwargs)
    elif file_extension in [".xlsx", ".xls"]:
        return pd.read_excel(path, **kwargs)
    else:
        raise ValueError(
            f"File extension {file_extension} not supported. Valid extensions are {VALID_EXTENSIONS}"
        )


# data = load_input_data("data/example/Data_test_salinity_caley.xlsx")
data = load_dataset("weldeab")
data

# add longitude and latitude
data["lat"] = 20.0
data["lon"] = 90.0

data
# %%
sigT = predict_sigmaT(data, "Poly2ESF", "d18Oc_mean", "d18Oc_stdev", "lon", "lat")
# %%
sigT = sigT.rename({"d18Oc": "age"}).assign_coords({"age": data.index.values})
# %%
quantiles = sigT.quantile([0.025, 0.25, 0.5, 0.75, 0.975], dim="sample")

# %%
density_woa18 = xr.open_dataset(
    "data/density/woa18_density_sigmaT_mean_1981-2010.nc"
).isel(depth=0)
density_woa18 = density_woa18.sel(lon=90, lat=19.9, method="nearest", drop=True)
density_woa18

# %%
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.fill_between(
    quantiles.age,
    quantiles.sel(quantile=0.025),
    quantiles.sel(quantile=0.975),
    color="k",
    alpha=0.1,
    label="95% CI",
)
ax.fill_between(
    quantiles.age,
    quantiles.sel(quantile=0.25),
    quantiles.sel(quantile=0.75),
    color="k",
    alpha=0.2,
    label="50% CI",
)
quantiles.sel(quantile=0.5).plot(
    x="age", ax=ax, color="darkblue", marker=".", ls="--", lw=1, label="Median"
)
ax.errorbar(
    0, density_woa18["mean"], yerr=density_woa18["stdev"], color="k", marker="o"
)
ax.set_xlabel("Age [yr BP]")
ax.set_ylabel("$\sigma_T$ [kg/m$^3$]")
ax.set_title("Predicted $\sigma_T$ for marine sediment core SO188-KL342 (Indian Ocean)")
ax.legend()

# %%


sigT_non_spatial = predict_sigmaT(
    data, "Poly2", "d18Oc_mean", "d18Oc_stdev", "lon", "lat"
)
sigT_non_spatial = sigT_non_spatial.rename({"d18Oc": "age"}).assign_coords(
    {"age": data.index.values}
)
quantiles_non_spatial = sigT_non_spatial.quantile(
    [0.025, 0.25, 0.5, 0.75, 0.975], dim="sample"
)

# %%
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
# Spatial model
ax.fill_between(
    quantiles.age,
    quantiles.sel(quantile=0.025),
    quantiles.sel(quantile=0.975),
    color="darkblue",
    alpha=0.1,
)
quantiles.sel(quantile=0.5).plot(
    x="age", ax=ax, color="darkblue", marker=".", ls="--", lw=1, label="Spatial model"
)
# Non-spatial model
ax.fill_between(
    quantiles_non_spatial.age,
    quantiles_non_spatial.sel(quantile=0.025),
    quantiles_non_spatial.sel(quantile=0.975),
    color="darkred",
    alpha=0.1,
)
quantiles_non_spatial.sel(quantile=0.5).plot(
    x="age",
    ax=ax,
    color="darkred",
    marker=".",
    ls="--",
    lw=1,
    label="Non-spatial model",
)
ax.errorbar(
    0,
    density_woa18["mean"],
    yerr=density_woa18["stdev"],
    color="k",
    marker="o",
    label="Today",
)
ax.set_xlabel("Age [yr BP]")
ax.set_ylabel("$\sigma_T$ [kg/m$^3$]")
ax.legend()
ax.set_title("Predicted $\sigma_T$ for marine sediment core SO188-KL342 (Indian Ocean)")

# %%
