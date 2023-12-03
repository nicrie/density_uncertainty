# %%
import xarray as xr
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


def get_nearest_sigmaT(data, sigmaT, lon_step=1, lat_step=1):
    """Match each data point with a sigma T value."""

    if lon_step != 1 or lat_step != 1:
        sigmaT = sigmaT.coarsen(lon=lon_step, lat=lat_step, boundary="trim").mean()

    sigmaT_matched = data.apply(
        lambda x: sigmaT["mu"].sel(lon=x.lon, lat=x.lat, method="nearest").item(),
        axis=1,
    )
    return sigmaT_matched


# %%
# Load data
# =============================================================================
# sigma T density from WOA18
sigmaT = xr.open_dataset("data/density/woa18_density_sigmaT_1981-2010.nc")


# %%
# d18Oc from Malevich et al. 2019
data = pd.read_excel("data/d18Oc/Malevich et al., 2019b.xlsx")
data
data = data[["latitude", "longitude", "species", "d18oc"]]
data.columns = ["lat", "lon", "species", "d18Oc"]
n_samples_all = data.shape[0]


# %%
# Match each d18Oc sample with a sigma T value
# -----------------------------------------------------------------------------
matched_sigmaT = get_nearest_sigmaT(data, sigmaT, lon_step=1, lat_step=1)
n_missing = matched_sigmaT.isnull().sum()
print("1: ", n_missing)
step = 2
while step <= 8 and n_missing > 0:
    # As long as there are missing sigma T values, increase the smoothing step
    temp = get_nearest_sigmaT(data, sigmaT, lon_step=step, lat_step=step)
    matched_sigmaT.fillna(temp, inplace=True)
    n_missing = matched_sigmaT.isnull().sum()
    print("{:}: {:}".format(step, n_missing))
    step += 1

data["sigmaT"] = matched_sigmaT
data.to_csv("data/d18Oc_sigmaT_database.csv", index=False)
# %%
