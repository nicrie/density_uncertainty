# %%
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from cartopy.crs import Robinson, PlateCarree
from cartopy.feature import LAND, OCEAN

# %%

data = pd.read_csv("data/d18Oc_sigmaT_database.csv")
# %%

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111, projection=Robinson())
ax.add_feature(LAND, color=".8")
ax.add_feature(OCEAN, color=".3")
ax.set_global()
sns.scatterplot(
    data=data, x="lon", y="lat", hue="species", ax=ax, s=10, transform=PlateCarree()
)
plt.legend(ncols=5, bbox_to_anchor=(0.5, 0.0), loc="upper center", frameon=False)
plt.title("Sample locations (n={})".format(len(data)))
plt.savefig("figs/database/forams_sample_locations.png", dpi=300, bbox_inches="tight")
plt.show()


# %%

fig = plt.figure(figsize=(12, 5))
gs = fig.add_gridspec(1, 1, width_ratios=[1])
ax = fig.add_subplot(gs[0], projection=Robinson())
ax.add_feature(LAND, color=".8")
ax.add_feature(OCEAN, color=".3")
ax.set_global()
sns.scatterplot(
    data=data,
    x="lon",
    y="lat",
    hue="d18Oc",
    palette="viridis",
    ax=ax,
    s=10,
    transform=PlateCarree(),
    legend=False,
)
plt.legend(ncols=5, bbox_to_anchor=(0.5, 0.0), loc="upper center", frameon=False)
plt.title("$\delta^{18}$O$_{c}$ (‰ VSMOW)")
plt.savefig("figs/database/forams_d18Oc.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
fig = plt.figure(figsize=(12, 5))
gs = fig.add_gridspec(1, 1, width_ratios=[1])
ax = fig.add_subplot(gs[0], projection=Robinson())
ax.add_feature(LAND, color=".8")
ax.add_feature(OCEAN, color=".3")
ax.set_global()
sns.scatterplot(
    data=data,
    x="lon",
    y="lat",
    hue="sigmaT",
    palette="viridis",
    ax=ax,
    s=10,
    transform=PlateCarree(),
    legend=False,
)
plt.legend(ncols=5, bbox_to_anchor=(0.5, 0.0), loc="upper center", frameon=False)
plt.title("$\sigma_{T}$ (kg m$^{-3}$)")
plt.savefig("figs/database/forams_sigma_T.png", dpi=300, bbox_inches="tight")
plt.show()


# %%
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111)
sns.scatterplot(data=data, x="d18Oc", y="sigmaT", hue="species", ax=ax, s=10)
ax.set_xlabel("d18Oc (‰)")
ax.set_ylabel("sigma T (kg m$^{-3}$)")
sns.despine(ax=ax, trim=True, offset=5)
plt.savefig("figs/database/d18Oc_vs_sigmaT.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
