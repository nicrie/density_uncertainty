# %%
import xarray as xr
import arviz as az
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import LAND
import pandas as pd


# %%

EXPERIMENTS = ["poly1_pool", "poly2_pool", "poly1_hier"]


idata = {}
reg_results = {}

for exp in EXPERIMENTS:
    idata[exp] = az.from_netcdf(f"data/regression/obs/{exp}/idata.nc")
    reg_results[exp] = pd.read_csv(f"data/regression/obs/{exp}/regression_results.csv")


# %%
# Create figures
# =============================================================================
for exp in EXPERIMENTS:
    # Trace plot (extended)
    az.plot_trace(
        idata[exp],
        filter_vars="regex",
        var_names=[
            "prior",
            "beta",
            "sigma",
        ],
        legend=True,
    )
    plt.savefig(
        "figs/regression/obs/trace_ext_{:}.png".format(exp),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # Trace plot (sigma only)
    ax = az.plot_trace(
        idata[exp],
        var_names=[
            "sigma",
        ],
        legend=True,
    )
    ax.flatten()[0].set_xlim(0.2, 1.0)
    plt.savefig(
        "figs/regression/obs/trace_sigma_{:}.png".format(exp),
        dpi=150,
        bbox_inches="tight",
    )
    plt.suptitle("experiment {:}".format(exp))
    plt.close()

    # Posterior predictive regression plot
    az.plot_lm(idata=idata[exp], y_hat="sigmaT", y="sigmaT", x="d18Oc")
    plt.ylim(17, 32)
    plt.xlim(-4, 5)
    plt.title("experiment {:}".format(exp))
    plt.savefig(
        "figs/regression/obs/posterior_predictive_regression_{:}.png".format(exp),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


# %%
# Model comparison
# -----------------------------------------------------------------------------
df_comp_loo = az.compare(
    {exp_name: idata for exp_name, idata in idata.items()},
)
df_comp_loo

az.plot_compare(df_comp_loo)
plt.savefig("figs/regression/obs/model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# Residual plot
# -----------------------------------------------------------------------------
trans_proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(7.2, 9), dpi=500)
ax1 = fig.add_subplot(3, 1, 1, projection=ccrs.Robinson())
ax2 = fig.add_subplot(3, 1, 2, projection=ccrs.Robinson())
ax3 = fig.add_subplot(3, 1, 3, projection=ccrs.Robinson())
axes = [ax1, ax2, ax3]
for ax in axes:
    ax.set_global()
    ax.coastlines(lw=0.2)
    ax.add_feature(LAND, facecolor=".8")

for i, exp in enumerate(EXPERIMENTS):
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
        ax=axes[i],
        transform=trans_proj,
    )
    axes[i].set_title(f"Residuals {exp}")
plt.savefig("figs/regression/obs/residuals.png", dpi=500, bbox_inches="tight")
plt.show()
# %%
