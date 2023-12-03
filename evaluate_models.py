# %%
import os
import warnings
from pprint import pprint

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
import xarray as xr
from cartopy.crs import PlateCarree, Robinson
from cartopy.feature import LAND, OCEAN
from esda.moran import Moran
from libpysal.weights import full2W
from sklearn.metrics import r2_score

import pymc_models
from utils.linalg import row_standardize

warnings.simplefilter("ignore")

# %%
# Load data
# =============================================================================
print("Loading data...")
# experiments = ["full", "ruber"]
experiments = ["full"]

data_ext = {}
weight_matrix = {}
for exp in experiments:
    data_ext[exp] = pd.read_csv(f"data/regression/{exp}/database_extended.csv")
    weight_matrix[exp] = pd.read_csv(f"data/esf/{exp}/C.csv")


# %%
# Load models
# =============================================================================
print("Loading models...")
model_types = {
    "Linear": pymc_models.Linear,
    "Poly2": pymc_models.Poly2,
    "LinearESF": pymc_models.LinearRHSESF,
    "Poly2ESF": pymc_models.Poly2RHSESF,
}
models = {}
idata = {}

for exp in experiments:
    models[exp] = {}
    idata[exp] = {}
    for name, model in model_types.items():
        dir_path = f"data/regression/{exp}/{name}/"
        try:
            file_path = os.path.join(dir_path, "model.nc")
            models[exp][name] = model.load(file_path)
        except FileNotFoundError:
            pass
        try:
            file_path = os.path.join(dir_path, "inference_data.nc")
            idata[exp][name] = az.from_netcdf(file_path)
        except FileNotFoundError:
            pass


# %%
# Trace plots
# =============================================================================
print("Plotting trace plots...")
for exp in experiments:
    for model_name, id in idata[exp].items():
        dir_path = f"figs/regression/{exp}/"
        fig_path = os.path.join(dir_path, f"trace_plot_{model_name}.png")
        os.makedirs(dir_path, exist_ok=True)

        var_names = list(id.posterior.data_vars.keys())
        if "x_est" in var_names:
            var_names.remove("x_est")
        az.plot_trace(
            id,
            var_names=var_names,
        )
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()


# %%
# LOO & WAIC
# =============================================================================
print("Computing LOO & WAIC...")


def create_fig_loo_waic(exp, df_loo, df_waic):
    fig = plt.figure(figsize=(14, 7))
    axes = [fig.add_subplot(1, 2, i) for i in range(1, 3)]
    az.plot_compare(df_loo, ax=axes[0])
    az.plot_compare(df_waic, ax=axes[1])
    axes[0].legend()
    axes[1].legend()
    fig.subplots_adjust(wspace=0.3, right=0.98)
    plt.savefig(f"figs/regression/{exp}/compare.png")
    plt.close()


loo = {}
waic = {}
for exp in experiments:
    loo[exp] = {}
    waic[exp] = {}
    for name, id in idata[exp].items():
        loo[exp][name] = az.loo(id)
        waic[exp][name] = az.waic(id)

    df_comp_loo = az.compare(loo[exp])
    df_comp_waic = az.compare(waic[exp])

    path_data = f"data/regression/{exp}/"
    path_figs = f"figs/regression/{exp}/"
    os.makedirs(os.path.dirname(path_data), exist_ok=True)
    os.makedirs(os.path.dirname(path_figs), exist_ok=True)
    df_comp_loo.to_csv(os.path.join(path_data, "loo.csv"))
    df_comp_waic.to_csv(os.path.join(path_data, "waic.csv"))

    create_fig_loo_waic(exp, df_comp_loo, df_comp_waic)

    pprint(df_comp_loo)

# %%
# Posterior predictive checks
# =============================================================================
print("Computing posterior predictive ...")
predictions = {}
quantiles = {}
qs = [0.025, 0.25, 0.75, 0.975]
for exp in experiments:
    predictions[exp] = {}
    quantiles[exp] = {}
    dext = data_ext[exp]
    for model_name, idt in idata[exp].items():
        preds = idt.posterior_predictive["sigmaT"]
        preds.coords.update({"d18Oc": dext["d18Oc"].values})
        predictions[exp][model_name] = preds
        quantiles[exp][model_name] = preds.quantile(qs, ("chain", "draw")).sortby(
            "d18Oc"
        )
        pred_means = preds.mean(("chain", "draw"))
        data_ext[exp][f"sigmaT_pred_{model_name}"] = pred_means
        data_ext[exp][f"sigmaT_res_{model_name}"] = dext["sigmaT"] - pred_means

    path = f"data/regression/{exp}/database_extended.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data_ext[exp].to_csv(path, index=False)

# %%
# Morans I statistic
# =============================================================================
print("Computing Moran's I statistic...")


def compute_morans(data, W, col):
    W = full2W(W, ids=list(data.index.values))
    gm = Moran(data[col].values, W, transformation="r", permutations=0)
    return gm.I, gm.z_norm, gm.p_norm


MCs = {}
for exp in experiments:
    MCs[exp] = {}
    dext = data_ext[exp]
    W = weight_matrix[exp].values
    W = row_standardize(W)
    for model_name in models[exp].keys():
        result = compute_morans(dext, W, f"sigmaT_res_{model_name}")
        print(f"{exp} | {model_name} | {result}")
        MCs[exp][model_name] = result


# %%
# R2 score
# =============================================================================

print("Computing R2 scores...")

r2_scores = {}
for exp in experiments:
    r2_scores[exp] = {}
    dext = data_ext[exp]
    for name, model in models[exp].items():
        r2_scores[exp][name] = r2_score(dext["sigmaT"], dext[f"sigmaT_pred_{name}"])
# %%
# Residuals map
# =============================================================================
print("Plotting residuals map...")
cmap = "icefire"
for exp in experiments:
    fig = plt.figure(figsize=(14, 7))
    axes = [fig.add_subplot(2, 2, i, projection=Robinson()) for i in range(1, 5)]
    for ax, m in zip(axes, models[exp].keys()):
        ax.add_feature(LAND, color=".8")
        ax.add_feature(OCEAN, color=".3")
        ax.set_global()
        data_ext[exp].plot.scatter(
            x="lon",
            y="lat",
            c="sigmaT_res_" + m,
            cmap=cmap,
            vmin=-1.5,
            vmax=1.5,
            ax=ax,
            marker=".",
            transform=PlateCarree(),
            colorbar=False,
        )
        ax.set_title(
            "{:} | $\sigma_M =${:.4f} | $R^2$ = {:.3}".format(
                m, MCs[exp][m][0], r2_scores[exp][m]
            )
        )
    plt.tight_layout()
    path = f"figs/regression/{exp}/residuals_map.png"
    plt.savefig(path, dpi=300)


# %%
# Prediction
# =============================================================================
print("Plotting prediction...")
for exp in experiments:
    dext = data_ext[exp].sort_values("d18Oc")

    fig = plt.figure(figsize=(14, 7))
    axes = [fig.add_subplot(2, 2, i) for i in range(1, 5)]
    for ax, m in zip(axes, models[exp].keys()):
        # Plot observations
        dext.plot(
            x="d18Oc",
            y="sigmaT",
            marker=".",
            linestyle="none",
            ax=ax,
            markersize=3,
            color=".7",
            alpha=0.75,
            legend=False,
        )
        dext.plot(x="d18Oc", y=f"sigmaT_pred_{m}", ax=ax, lw=0.5, legend=False)
        # Add prediction quantiles
        quants = quantiles[exp][m]
        ax.fill_between(
            quants["d18Oc"],
            quants.sel(quantile=0.025),
            quants.sel(quantile=0.975),
            color="k",
            alpha=0.05,
        )
        ax.fill_between(
            quants["d18Oc"],
            quants.sel(quantile=0.25),
            quants.sel(quantile=0.75),
            color="k",
            alpha=0.1,
        )

        ax.set_title(m)
        ax.set_xlabel("$\delta^{18}O_c$ (‰ VSMOW)")
        ax.set_ylabel("$\sigma_T$ (kg m$^{-3}$)")
        sns.despine(ax=ax, trim=True, offset=5)

    plt.tight_layout()
    path = f"figs/regression/{exp}/prediction.png"
    plt.savefig(path, dpi=300)


# %%
