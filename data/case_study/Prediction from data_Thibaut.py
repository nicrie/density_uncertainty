
import arviz as az
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import nutpie
import pandas as pd
import preliz as pz
import pymc as pm
import os
import regionmask as rm
import xarray as xr
import xarray_regrid
from cartopy.feature import LAND
from pymc import Exponential, Gamma, Normal

from utils.constants import MAP_SPECIES_IDX




# Load new independent dataset (contains d18Oc for which sigmaT needs to be predicted)
new_data = pd.read_csv("d18O_LH_Database.txt", sep="\t", decimal=",")
new_data["species_idx"] = new_data["species"].map(MAP_SPECIES_IDX)  # Map species to indices
new_data = new_data.dropna()  # Drop rows with missing values
d18Oc_new = new_data["d18Oc"].values  # Extract the new d18Oc values

# Load saved Bayesian model results
idata = az.from_netcdf("data/regression/obs/poly1_hier/idata.nc")

# Match the dimensions of priors extracted from idata (remove unnecessary dimensions)
beta0_mu = idata.prior["prior_beta0_mu"].values.squeeze()
beta0_sigma = idata.prior["prior_beta0_sigma"].values.squeeze()
beta1_mu = idata.prior["prior_beta1_mu"].values.squeeze()
beta1_sigma = idata.prior["prior_beta1_sigma"].values.squeeze()
sigma_values = idata.prior["prior_std"].values.squeeze()

# Validate the shapes
print("beta0_mu shape:", beta0_mu.shape)  # Should match the number of species
print("beta1_mu shape:", beta1_mu.shape)
print("sigma_values shape:", sigma_values.shape)
print("species_idx shape:", new_data["species_idx"].shape)  # Matches number of rows in new_data
print("d18Oc shape:", d18Oc_new.shape)

# Recreate the Bayesian model for prediction
with pm.Model(coords={
    "d18Oc": d18Oc_new,
    "species": list(MAP_SPECIES_IDX.keys())  # Keys correspond to the number of species
}) as model:
    # Define data inputs for the new dataset
    x = pm.Data("x", d18Oc_new, dims=("d18Oc",))
    species_idx = pm.Data("species_idx", new_data["species_idx"].values, dims=("d18Oc",))

    # Redefine priors using stored parameters
    beta0 = pm.Normal("beta0", mu=beta0_mu, sigma=beta0_sigma, dims="species")
    beta1 = pm.Normal("beta1", mu=beta1_mu, sigma=beta1_sigma, dims="species")
    sigma = pm.Exponential("sigma", lam=1 / (sigma_values ** 2), dims="species")

    # Define deterministic relationship and likelihood
    mu = pm.Deterministic("mu", beta0[species_idx] + beta1[species_idx] * x, dims=("d18Oc",))
    sigmaT = pm.Normal("sigmaT", mu=mu, sigma=sigma[species_idx], dims=("d18Oc",))

    # Load new data into the model
    pm.set_data({"x": d18Oc_new, "species_idx": new_data["species_idx"].values})

    # Generate posterior predictive samples for sigmaT
    posterior_predictive = pm.sample_posterior_predictive(
        idata, var_names=["sigmaT"], extend_inferencedata=False
    )


# Extract sigmaT directly from posterior_predictive.posterior_predictive
sigmaT = posterior_predictive.posterior_predictive["sigmaT"]

# Calculate the mean and standard deviation over chain and draw dimensions
predicted_sigmaT = sigmaT.mean(dim=("chain", "draw")).values  # Mean prediction
prediction_error = sigmaT.std(dim=("chain", "draw")).values   # Prediction uncertainty (std)

# Validate shapes before adding to DataFrame
print("predicted_sigmaT shape:", predicted_sigmaT.shape)  # Should match the length of new_data
print("prediction_error shape:", prediction_error.shape)
print("new_data shape:", new_data.shape)  # Ensure alignment with new_data

# Add predictions and uncertainties to the DataFrame
new_data["predicted_sigmaT"] = predicted_sigmaT
new_data["error_sigmaT"] = prediction_error

# Save the updated DataFrame to a CSV file
new_data.to_csv("predictions_sigmaT_new_data.csv", index=False)
print("Predictions saved to 'predictions_sigmaT_new_data.csv'.")





