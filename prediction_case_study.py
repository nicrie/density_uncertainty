# %%
import xarray as xr
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd


# Vectorized prediction function
def predict_sigmaT_vectorized(d18O, species, beta0, beta1):
    """
    Make density predictions in a vectorized way using xarray broadcasting

    Parameters:
    -----------
    d18O : array-like
        Oxygen isotope values
    species : array-like
        Species names matching those in beta0 and beta1
    beta0, beta1 : xr.DataArray
        Model parameters with dimensions (chain, draw, species)

    Returns:
    --------
    xr.DataArray
        Predictions with dimensions (chain, draw, sample)
    """
    # Create input DataArrays
    d18O_da = xr.DataArray(d18O, dims="sample")
    species_da = xr.DataArray(
        species, dims="sample", coords={"sample": range(len(species))}
    )

    # Select the appropriate beta parameters for each species
    b0 = beta0.sel(species=species_da)
    b1 = beta1.sel(species=species_da)

    # Apply the linear model - broadcasting happens automatically
    predictions = b0 + b1 * d18O_da

    return predictions


# %%
# Load data
# =============================================================================
model = az.from_netcdf("data/regression/obs/poly1_hier/idata.nc")
beta0 = model.posterior.beta0
beta1 = model.posterior.beta1

test_data = pd.read_csv(
    "data/case_study/d18O_LH_Database.txt",
    sep="\t",
    converters={0: lambda x: float(x.replace(",", "."))},
)

# %%
# Use model to make predictions
# =============================================================================
# Apply the prediction function to test data
test_species = test_data["species"].values
test_d18O = test_data["d18Oc"].values

# Make predictions
predictions = predict_sigmaT_vectorized(test_d18O, test_species, beta0, beta1)

# Calculate summary statistics
pred_mean = predictions.mean(dim=["chain", "draw"])
pred_ci_low = predictions.quantile(0.025, dim=["chain", "draw"])
pred_ci_high = predictions.quantile(0.975, dim=["chain", "draw"])

# Create a summary dataframe
results = pd.DataFrame(
    {
        "d18Oc": test_d18O,
        "species": test_species,
        "sigmaT_mean": pred_mean.values,
        "sigmaT_ci_low": pred_ci_low.values,
        "sigmaT_ci_high": pred_ci_high.values,
    }
)

# Display results
results

# %%
