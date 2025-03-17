# %%
"""
Preprocess climate simulation data: Combine the data from the different climate simulations into a individual datasets.
"""

import xarray as xr

root = "data/simulations/"

# Breitkreuz et al. 2018
bk_d18Oc = xr.open_dataarray(root + "breitkreuz/d18Oc_mean_annual_weighted.nc")
bk_density = xr.open_dataarray(root + "breitkreuz/surface_density_in_situ_mean.nc")
breitkreuz = xr.Dataset({"d18Oc": bk_d18Oc, "sigmaT": bk_density - 1000})


# iLOVECLIM
ilc_d18Oc_LGM = xr.open_dataarray(root + "iLOVECLIM/d18Oc_mean_LGM_iLOVECLIM.nc")
ilc_d18Oc_PI = xr.open_dataarray(root + "iLOVECLIM/d18Oc_mean_PI_iLOVECLIM.nc")
ilc_density_LGM = xr.open_dataarray(
    root + "iLOVECLIM/surface_density_in_situ_mean_LGM_iLOVECLIM.nc"
)
ilc_density_PI = xr.open_dataarray(
    root + "iLOVECLIM/surface_density_in_situ_mean_PI_iLOVECLIM.nc"
)

ilc_d18Oc_LGM = ilc_d18Oc_LGM.expand_dims({"time": ["LGM"]})
ilc_d18Oc_PI = ilc_d18Oc_PI.expand_dims({"time": ["PI"]})
ilc_d18Oc = xr.concat([ilc_d18Oc_LGM, ilc_d18Oc_PI], dim="time")

ilc_density_LGM = ilc_density_LGM.expand_dims({"time": ["LGM"]})
ilc_density_PI = ilc_density_PI.expand_dims({"time": ["PI"]})
ilc_density = xr.concat([ilc_density_LGM, ilc_density_PI], dim="time")

iloveclim = xr.Dataset({"d18Oc": ilc_d18Oc, "sigmaT": ilc_density - 1000})

# ECHAM-MPI
mpi_d18Oc_LGM = xr.open_dataarray(
    root + "ECHAM-MPI/d18Oc_mean_annual_weighted_LGM_ECHAM_MPI.nc"
)
mpi_d18Oc_PI = xr.open_dataarray(
    root + "ECHAM-MPI/d18Oc_mean_annual_weighted_PI_ECHAM_MPI.nc"
)
mpi_density_LGM = xr.open_dataarray(
    root + "ECHAM-MPI/surface_density_in_situ_mean_LGM_ECHAM_MPI.nc"
)
mpi_density_PI = xr.open_dataarray(
    root + "ECHAM-MPI/surface_density_in_situ_mean_PI_ECHAM_MPI.nc"
)

mpi_d18Oc_LGM = mpi_d18Oc_LGM.expand_dims({"time": ["LGM"]})
mpi_d18Oc_PI = mpi_d18Oc_PI.expand_dims({"time": ["PI"]})
mpi_d18Oc = xr.concat([mpi_d18Oc_LGM, mpi_d18Oc_PI], dim="time")

mpi_density_LGM = mpi_density_LGM.expand_dims({"time": ["LGM"]})
mpi_density_PI = mpi_density_PI.expand_dims({"time": ["PI"]})
mpi_density = xr.concat([mpi_density_LGM, mpi_density_PI], dim="time")

echam_mpi = xr.Dataset({"d18Oc": mpi_d18Oc, "sigmaT": mpi_density - 1000})

# %%

sim = xr.DataTree.from_dict(
    {
        "/": xr.Dataset(
            coords={"species": breitkreuz.species.values},
        ),
        "/historical": breitkreuz,
        "/paleo/": xr.Dataset(
            coords={"time": ["LGM", "PI"]},
        ),
        "/paleo/iloveclim": iloveclim,
        "/paleo/echam_mpi": echam_mpi,
    },
)
sim.to_netcdf(root + "simulations.nc", engine="netcdf4")

# %%
