# %%
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import trange

# %%
# Load sea level change by Spratt et al. (2016)
sealevel = pd.read_csv('data/sealevel/spratt2016.txt', skiprows=95, sep='\t', na_values='NaN', encoding='cp1252')
sealevel = sealevel[['age_calkaBP', 'SeaLev_longPC1', 'SeaLev_longPC1_err_sig']]

# We convert sea level change (SLC) to d18Ow using linear scaling.
# We assume the following values:
# Salinity ocean water = +0.0 at 0 kyr BP
# Salinity = +0.90 permil at -125 m (between 22 and 24 kyr BP in Spratt et al. (2016))
# This yields the following scaling equation
def sea_level_to_salinity(z_mean, z_stdev, seed=None):
    # S = a * z + b
    # a = S_lgm / (z_lgm - z_0)
    # b = S_lgm * z_0 / (z_lgm - z_0)
    # ==> S = S_lgm * (z - z_0) / (z_lgm - z_0)
    rng = np.random.default_rng(seed)
    S_lgm = rng.normal(0.90, 0.05, size=z_mean.size)
    z_lgm = rng.normal(-125, 5, size=z_mean.size)
    z0 = rng.normal(z_mean.iloc[0], z_stdev.iloc[0], size=z_mean.size)
    z = rng.normal(z_mean, z_stdev)

    return S_lgm * (z - z0) / (z_lgm - z0)

mean_sealevel = sealevel['SeaLev_longPC1']
stdev_sealevel = sealevel['SeaLev_longPC1_err_sig']

sals = sea_level_to_salinity(mean_sealevel, stdev_sealevel)

# %%
bst_sals = []
for n in trange(4000):
    bst_sals.append(sea_level_to_salinity(mean_sealevel, stdev_sealevel, n))

bst_sals = np.array(bst_sals)
bst_sals = xr.DataArray(
    bst_sals,
    dims=['draw', 'age'],
    coords={'draw': range(bst_sals.shape[0]), 'age': mean_sealevel.index},
    name='salinity',
)
# %%
# Save to netCDF
bst_sals.to_netcdf('data/salinity/global_salinity_from_sealevel.nc')


# %%
