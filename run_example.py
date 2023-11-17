# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
import utils.visualization as vis
from tqdm import trange


from utils.physics import salinity_approx
from importlib import reload

reload(vis)


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


# %%
# Global d18Ow derived from sea level
# -----------------------------------------------------------------------------
global_d18Ow = xr.open_dataarray("data/d18Ow/global_d18Ow_from_sealevel.nc")
global_d18Ow = global_d18Ow.sel(draw=slice(0, 2000), age=slice(0, 50))
global_d18Ow["age"] = global_d18Ow["age"] * 1000
new_ages = np.arange(0, 50000, 1)
global_d18Ow = global_d18Ow.interp_like(
    xr.DataArray(new_ages, dims="age", coords={"age": new_ages})
)

# %%
# Global salinity derived from sea level
# -----------------------------------------------------------------------------
global_salinity = xr.open_dataarray("data/salinity/global_salinity_from_sealevel.nc")
global_salinity = global_salinity.sel(draw=slice(0, 2000), age=slice(0, 50))
global_salinity["age"] = global_salinity["age"] * 1000
global_salinity = global_salinity.interp_like(
    xr.DataArray(new_ages, dims="age", coords={"age": new_ages})
)


# %%
# Load example data
# =============================================================================
from utils.datasets import load_dataset
from models import predict_sigmaT

example_data = load_dataset("rieger")
example_data.head()

N_DRAWS = 1000

data = example_data.copy()

vis.plot_data(
    data,
    "d18Oc_mean",
    "d18Oc_stdev",
    figname="md04_input_d18Oc",
    title="$\delta^{18}O_c$ for MD04-2875 (Arabian Sea)",
    y_inverse=True,
)

vis.plot_data(
    data,
    "T_mean",
    "T_stdev",
    figname="md04_input_temp",
    title="Temperature for MD04-2875 (Arabian Sea)",
)

# Correct d18Oc for d18Ow
# -----------------------------------------------------------------------------
data["d18Oc_mean"] = data["d18Oc_mean"] - global_d18Ow.sel(age=example_data.index).mean(
    "draw"
)
data["d18Oc_stdev"] = np.sqrt(
    data["d18Oc_stdev"] ** 2 + global_d18Ow.sel(age=example_data.index).var("draw")
)
data["lat"] = 25.0
data["lon"] = 64.0

# %%
# Predict sigmaT
# -----------------------------------------------------------------------------
model_types = ["Linear", "Poly2", "LinearESF", "Poly2ESF"]

sigmaT = {}
for model_type in model_types:
    sT = predict_sigmaT(
        data,
        model=model_type,
        name_d18Oc="d18Oc_mean",
        name_d18Oc_stdev="d18Oc_stdev",
        name_lon="lon",
        name_lat="lat",
    )
    sT = (
        sT.drop("sample")
        .assign_coords({"sample": range(sT.sample.size)})
        .rename({"sample": "draw"})
    )
    step = sT.draw.size // N_DRAWS
    sT = sT.isel(draw=slice(None, None, step))
    sT = sT.assign_coords({"draw": range(N_DRAWS)})
    sT = sT.rename({"d18Oc": "age"}).assign_coords({"age": data.index})
    sigmaT[model_type] = sT

# %%
# Plot predicted sigmaT
# -----------------------------------------------------------------------------
for model_type, sT in sigmaT.items():
    vis.plot_data_samples(
        sT, "draw", figname=f"md04_sigmaT_{model_type}", title=model_type, ylim=(20, 26)
    )

# %%
# Sample temperature
# -----------------------------------------------------------------------------
damean = data["T_mean"].to_xarray()
dastdev = data["T_stdev"].to_xarray()
temperature = draw_samples(damean, dastdev, N_DRAWS, "age")
# in Kelvin
temperature += 273.15


# %%
# Convert density to salinity
# -----------------------------------------------------------------------------
salinity = {}
salinity_local = {}

for model_type, sT in sigmaT.items():
    rho = 1000 + sT
    sal = salinity_approx(rho=rho, T=temperature, P=0.5)
    sal_local = sal - global_salinity

    salinity[model_type] = sal
    salinity_local[model_type] = sal_local

# %%

# %%
# Plot salinity
# -----------------------------------------------------------------------------
for model_type, sal in salinity.items():
    vis.plot_data_samples(
        sal,
        "draw",
        figname=f"md04_salinity_{model_type}",
        title=model_type,
        ylim=(30, 40),
    )
    vis.plot_data_samples(
        salinity_local[model_type],
        "draw",
        figname=f"md04_salinity_local_{model_type}",
        title=model_type,
        ylim=(30, 40),
    )

# %%
