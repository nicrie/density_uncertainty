# %%
import xarray as xr
import xarray_regrid

import arviz as az
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import nutpie
import pandas as pd
import preliz as pz
import pymc as pm
import regionmask as rm
import xarray as xr
import xarray_regrid
from cartopy.feature import LAND
from pymc import Exponential, Gamma, Normal


def fill_missing(da, copy=True):
    from scipy.interpolate import NearestNDInterpolator

    if copy:
        da_filled = da.copy(deep=True)
    else:
        da_filled = da

    indices = np.where(np.isfinite(da))
    interp = NearestNDInterpolator(np.transpose(indices), da.data[indices])
    da_filled[...] = interp(*np.indices(da.shape))
    return da_filled


# Set some meta information
# =============================================================================
translate_species = {
    "G. ruber": "ruber",
    "T. sacculifer": "sacculifer",
    "G. bulloides": "bulloides",
    "N. pachyderma": "pachy_s",
    "N. incompta": "pachy_d",
}

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

trans_proj = ccrs.PlateCarree()


# %%
# Load data
# =============================================================================
root_sim = "data/simulations/"
root_obs = "data/observations/"

# Observations
# -----------------------------------------------------------------------------
d18Oc_pts = pd.read_excel(root_obs + "d18Oc/Malevich et al., 2019b.xlsx")
d18Oc = xr.open_dataarray(root_obs + "d18Oc/malevich_d18Oc_1x1.nc")

# Rename species in d18Oc coordinate using translate_species dict
species = d18Oc.species.to_series().replace(translate_species).values
d18Oc = d18Oc.assign_coords(species=species)

sigmaT = xr.open_dataset(root_obs + "density/cmems_density_sigmaT_1993-2002.nc")
sigmaT = sigmaT.rename({"longitude": "lon", "latitude": "lat"})
sigmaT = sigmaT["mu"].regrid.linear(d18Oc)
sigmaT = fill_missing(sigmaT)

# Create a dataset with the sigmaT and d18Oc data
obs = xr.Dataset({"sigmaT": sigmaT, "d18Oc": d18Oc})
has_obs_d18Oc = obs.notnull().d18Oc
has_obs_sigmaT = has_obs_d18Oc.any("species")
has_obs = xr.Dataset({"d18Oc": has_obs_d18Oc, "sigmaT": has_obs_sigmaT})

# Simulations
# -----------------------------------------------------------------------------
sim = xr.open_datatree("data/simulations/simulations.nc")

# Only select the species present in the observations
sim = sim.sel(species=species)

# regrid the simulations to the same grid as the observations
iloveclim = sim["paleo/iloveclim"].to_dataset().regrid.linear(obs["sigmaT"])
echam_mpi = sim["paleo/echam_mpi"].to_dataset().regrid.linear(obs["sigmaT"])


iloveclim = iloveclim.where(has_obs)
echam_mpi = echam_mpi.where(has_obs)


data_prep = xr.DataTree.from_dict(
    {
        "/": obs,
        "/simulations": xr.Dataset(coords={"time": sim.paleo.time.values}),
        "/simulations/iloveclim": iloveclim,
        "/simulations/echam_mpi": echam_mpi,
    }
)
data_prep.to_netcdf("data/regression/data_prep.nc")
