# %%
import xarray as xr
from tqdm import tqdm

# %%
# Download ocean density (sigma) from World Ocean Atlas 2018
# =============================================================================
# Source: https://www.ncei.noaa.gov/access/world-ocean-atlas-2018/bin/woa18.pl
# Data: Mean of density on 1/4º grid for 1981-2010
# We need the variables for the mean and standard deviation of sigma
# I_an: Objectively analyzed mean fields for sea_water_sigma at standard depth levels.
# I_sd: The standard deviation about the statistical mean of sea_water_sigma in each grid-square at each standard depth level.
monthts = range(1, 13)
path = "https://www.ncei.noaa.gov/thredds-ocean/dodsC/ncei/woa/density/decav81B0/0.25/woa18_decav81B0_I{:02}_04.nc"
data = []
for month in tqdm(monthts, desc="Downloading WOA18"):
    ds = xr.open_dataset(path.format(month), decode_times=False)
    # Objectively analyzed mean fields for sea_water_sigma
    ds = ds["I_an"]
    # Select the surface layer
    ds = ds.sel(depth=0, drop=True)
    # Remove redundant dimensions
    ds = ds.squeeze(drop=True)
    # Downadload the data
    ds = ds.load()
    data.append(ds)
# %%
woa18 = xr.concat(data, dim="month")
woa18 = woa18.assign_coords({"month": monthts})
# %%
mu = woa18.mean("month")
sigma = woa18.std("month", ddof=1)
sigma.name = "sigmaT_standard_deviation"

density = xr.Dataset(
    {"mu": mu, "sigma": sigma},
    attrs={"description": "WOA18 surface water density (sigma T)", "units": "kg m-3"},
)


# %%
# Save to netcdf
density.to_netcdf("data/density/woa18_density_sigmaT_1981-2010.nc")
# %%
