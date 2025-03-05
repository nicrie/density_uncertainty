# %%
# cmems_obs-mob_glo_phy-sss_my_multi_P1M
# Dataset URL: https://data.marine.copernicus.eu/product/MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013/description
# Dataset manual: https://catalogue.marine.copernicus.eu/documents/PUM/CMEMS-MOB-PUM-015-013.pdf
import copernicusmarine
import xarray as xr

# %%
dataset_id = "cmems_obs-mob_glo_phy-sss_my_multi_P1M"
for year in range(1993, 2003):
    copernicusmarine.subset(
        dataset_id=dataset_id,
        force_download=True,
        output_directory="data/density/cmems/",
        variables=["dos"],
        start_datetime=f"{year}-01-01",
        end_datetime=f"{year}-12-31",
        output_filename=f"dataset-sss-ssd-rep-monthly_{year}.nc",
    )


# %%


data = xr.open_mfdataset("data/density/cmems/*.nc")
sigmaT = data["dos"].squeeze(drop=True) - 1000

mu = sigmaT.mean("time")
sigma = sigmaT.std("time")

import dask

mu, sigma = dask.compute(mu, sigma)
# %%
mu.plot()
sigmaT.sel(longitude=-13, latitude=-4, method="nearest").plot()
sigmaT.sel(longitude=8, latitude=3, method="nearest").plot()
# %%
mu.plot()
sigma.plot(robust=True)
(sigma / mu).plot(robust=True)

ds = xr.Dataset({"mu": mu, "sigma": sigma})
ds.to_netcdf("data/density/cmems_density_sigmaT_1993-2002.nc")
# %%
