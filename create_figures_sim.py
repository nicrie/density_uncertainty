# %%
import xarray as xr
import arviz as az
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import LAND
import pandas as pd

from utils.constants import EXPERIMENTS

# %%

idata = {}
reg_results = {}
reg_results_lgm = {}

for exp in EXPERIMENTS:
    path = f"data/regression/sim/{exp}/"
    idata[exp] = az.from_netcdf(path + "idata.nc")
    reg_results[exp] = pd.read_csv(path + "regression_results.csv")
    reg_results_lgm[exp] = pd.read_csv(path + "regression_results_test.csv")


# %%
# Create figures
# =============================================================================
for exp in EXPERIMENTS:
    # Trace plot (extended)
    az.plot_trace(
        idata[exp],
        var_names=[
            "prior_intercept_mu",
            "prior_intercept_sigma",
            "beta0",
            "prior_slope_mu",
            "prior_slope_sigma",
            "beta1",
            "prior_std",
            "sigma",
        ],
        legend=True,
    )
    plt.savefig(
        "figs/regression/sim/trace_ext_{:}.png".format(exp),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # Trace plot (sigma only)
    az.plot_trace(
        idata[exp],
        var_names=[
            "prior_std",
            "sigma",
        ],
        legend=True,
    )
    plt.savefig(
        "figs/regression/sim/trace_sigma_{:}.png".format(exp),
        dpi=150,
        bbox_inches="tight",
    )
    plt.suptitle("experiment {:}".format(exp))
    plt.close()

    # Posterior predictive regression plot
    axes = az.plot_lm(idata=idata[exp], y_hat="sigmaT", y="sigmaT", x="d18Oc")
    plt.ylim(17, 32)
    plt.xlim(-3, 5)
    plt.title("experiment {:}".format(exp))
    reg_results_lgm[exp].plot.scatter(
        x="d18Oc", y="sigmaT", ax=axes[0][0], zorder=500, s=5, label="LGM"
    )
    plt.savefig(
        "figs/regression/sim/posterior_predictive_regression_{:}.png".format(exp),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # Residuals plot (train; PI)
    trans_proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(7.2, 3), dpi=500)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines(lw=0.2)
    ax.add_feature(LAND, facecolor=".8")
    reg_results[exp].plot.scatter(
        x="lon",
        y="lat",
        c="residuals",
        cmap="RdBu",
        s=10,
        ec=".3",
        lw=0.2,
        vmin=-2,
        vmax=2,
        alpha=0.7,
        ax=ax,
        transform=trans_proj,
    )
    ax.set_title("Residuals {:} (train; PI)".format(exp))
    plt.savefig(
        "figs/regression/sim/residuals_{:}.png".format(exp),
        dpi=500,
        bbox_inches="tight",
    )
    plt.close()

    # Residuals plot (test; LGM)
    trans_proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(7.2, 3), dpi=500)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines(lw=0.2)
    ax.add_feature(LAND, facecolor=".8")
    reg_results_lgm[exp].plot.scatter(
        x="lon",
        y="lat",
        c="residuals",
        cmap="RdBu",
        s=10,
        ec=".3",
        lw=0.2,
        vmin=-2,
        vmax=2,
        alpha=0.7,
        ax=ax,
        transform=trans_proj,
    )
    ax.set_title("Residuals {:} (test; LGM)".format(exp))
    plt.savefig(
        "figs/regression/sim/residuals_{:}_lgm.png".format(exp),
        dpi=500,
        bbox_inches="tight",
    )
    plt.close()

# %%
