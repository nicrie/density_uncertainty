# %%
import numpy as np
import xarray as xr
import pandas as pd

from tqdm import trange


# %%
# Load sea level change by Spratt et al. (2016)
sealevel = pd.read_csv('data/sealevel/spratt2016.txt', skiprows=95, sep='\t', na_values='NaN', encoding='cp1252')
sealevel = sealevel[['age_calkaBP', 'SeaLev_longPC1', 'SeaLev_longPC1_err_sig']]

# We convert sea level change (SLC) to d18Ow using linear scaling.
# We assume the following values:
# d18Owater = +0.0 permil at 0 kyr BP
# d18Owater = +1.05 permil at -125 m (between 22 and 24 kyr BP in Spratt et al. (2016))
# This yields the following scaling equation
def sea_level_to_d18Ow(z_mean, z_stdev, seed=None):
    # d18Ow = a * z + b
    # a = (d18Ow_lgm) / (z_lgm - z_0)
    # b = - d18Ow_lgm / (z_lgm - z_0)
    # ==> d18Ow = d18Ow_lgm * (z - z_0) / (z_lgm - z_0)
    # d18Ow_lgm = 1.05 +- 0.05
    # z_lgm = -125 +- 5
    # z_0 = z[0]
    rng = np.random.default_rng(seed)
    d18Ow_lgm = rng.normal(1.05, 0.05, size=z_mean.size)
    z_lgm = rng.normal(-125, 5, size=z_mean.size)
    z0 = rng.normal(z_mean.iloc[0], z_stdev.iloc[0], size=z_mean.size)
    z = rng.normal(z_mean, z_stdev)

    return d18Ow_lgm * (z - z0) / (z_lgm - z0)

mean_sealevel = sealevel['SeaLev_longPC1']
stdev_sealevel = sealevel['SeaLev_longPC1_err_sig']

# %%
d18Ow = []
for n in trange(4000):
    d18Ow.append(sea_level_to_d18Ow(mean_sealevel, stdev_sealevel, n))

d18Ow = np.array(d18Ow)
d18Ow = xr.DataArray(
    d18Ow,
    dims=['draw', 'age'],
    coords={'draw': range(d18Ow.shape[0]), 'age': mean_sealevel.index},
    name='d18Ow',
)

# Save to netCDF
d18Ow.to_netcdf('data/d18Ow/global_d18Ow_from_sealevel.nc')
# %%

