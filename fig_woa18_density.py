# %%
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean.cm as cmo

sws = xr.open_dataset("data/density/woa18_density_sigmaT_1981-2010.nc")

# Coefficient of variation
sws["cv"] = sws["sigma"] / sws["mu"] * 100

fig, axes = plt.subplots(
    1, 3, figsize=(12, 3), subplot_kw={"projection": ccrs.Robinson()}
)
for ax in axes:
    ax.add_feature(cfeature.LAND, zorder=0, color=".2")
    ax.add_feature(cfeature.OCEAN, zorder=1, color=".4")
    ax.set_global()

sws["mu"].plot(
    ax=axes[0],
    transform=ccrs.PlateCarree(),
    cmap=cmo.dense,
    cbar_kwargs={
        "pad": 0.05,
        "orientation": "horizontal",
        "label": "Density $\sigma_T$ (kg/m^3)",
    },
    robust=True,
    zorder=2,
)
sws["sigma"].plot(
    ax=axes[1],
    transform=ccrs.PlateCarree(),
    cmap="viridis",
    cbar_kwargs={
        "pad": 0.05,
        "orientation": "horizontal",
        "label": "Standard deviation of $\sigma_T$ (kg/m^3)",
    },
    robust=True,
    zorder=2,
)
sws["cv"].plot(
    ax=axes[2],
    transform=ccrs.PlateCarree(),
    cmap="viridis",
    cbar_kwargs={
        "pad": 0.05,
        "orientation": "horizontal",
        "label": "Coefficient of variation (%)",
    },
    robust=True,
    zorder=2,
)
for ax in axes:
    ax.set_title("")
fig.subplots_adjust(left=0.05, right=0.95, top=1)
fig.suptitle(f"WOA18 $\sigma_T$ at {0} m")
plt.savefig("figs/density/woa18/density_sigma0_{:.0f}.png".format(0), dpi=300)

# %%
