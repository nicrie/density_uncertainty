import numpy as np
import pandas as pd
import xarray as xr

# Load your CSV file; adjust the filename and column names as needed.
df = pd.read_excel("data/d18Oc/Malevich et al., 2019b.xlsx")

df = df[["latitude", "longitude", "species", "d18oc"]]
df.loc[:, "species"] = df["species"].str.replace(",", ".")

# Extract the coordinate and value arrays.
lon = df["longitude"].values
lat = df["latitude"].values
val = df["d18oc"].values
species = df["species"].values

SPECIES = df.species.unique()

# Define bin edges for 1° resolution.
# For longitude from -180° to 180° and latitude from -90° to 90°:
lon_bins = np.arange(-180, 181, 1)  # edges: -180, -179, …, 180
lat_bins = np.arange(-90, 91, 1)  # edges: -90, -89, …, 90

mean_grids = []
for spc in SPECIES:
    # Filter the data for the current species.
    mask = np.isin(species, spc)
    lon_spc = lon[mask]
    lat_spc = lat[mask]
    val_spc = val[mask]

    # Compute the sum of values in each grid cell.
    sum_grid, _, _ = np.histogram2d(
        lat_spc, lon_spc, bins=[lat_bins, lon_bins], weights=val_spc
    )

    # Compute the count of measurements in each grid cell.
    count_grid, _, _ = np.histogram2d(lat_spc, lon_spc, bins=[lat_bins, lon_bins])

    # Compute the mean, handling division by zero by using np.divide with 'where'.
    mean_grid = np.divide(sum_grid, count_grid, where=(count_grid > 0))
    mean_grid[count_grid == 0] = np.nan  # Set grid cells with no measurements to nan

    # 'mean_grid' is a 2D array where rows correspond to latitude bins and columns to longitude bins.
    mean_grids.append(mean_grid)


# Conver to DataArray

mean_grids = np.stack(mean_grids, axis=-1)

# Create a DataArray with the mean grids and the corresponding coordinates.
da = xr.DataArray(
    mean_grids,
    coords={
        "lat": lat_bins[:-1] + 0.5,
        "lon": lon_bins[:-1] + 0.5,
        "species": SPECIES,
    },
    dims=["lat", "lon", "species"],
    name="d18Oc",
)
da.attrs["description"] = (
    "Oxygen isotopic composition of plantic foraminiferal calcite from core tops."
)
da.attrs["units"] = "permil"
da.attrs["long_name"] = "d18Oc"
da.attrs["reference"] = "Malevich et al. 2019"
da.attrs["doi"] = "10.1029/2019PA003576"

da.to_netcdf("data/d18Oc/malevich_d18Oc_1x1.nc")
