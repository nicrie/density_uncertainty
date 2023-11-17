# %%
import xarray as xr
import arviz as az

# %%
# Load inference results
# =============================================================================
idata = az.from_netcdf("data/regression/inference_data.nc")

# %%
az.plot_trace(
    trace_poly2_nonspatial,
    var_names=["beta0", "beta1", "beta2", "sigma_y"],
    combined=True,
)
