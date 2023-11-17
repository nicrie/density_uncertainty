# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from cartopy.crs import PlateCarree, Robinson
from cartopy.feature import LAND, OCEAN

# %%
data = pd.read_csv("data/esf/database_clustered.csv")
centroids = pd.read_csv("data/esf/centroids.csv")
n_clusters = centroids.shape[0]

# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection=Robinson())
ax.add_feature(LAND, color=".8")
ax.add_feature(OCEAN, color=".3")
ax.set_global()
sns.scatterplot(
    data=data,
    x="lon",
    y="lat",
    marker=".",
    edgecolor="None",
    hue="cluster",
    palette="tab20",
    s=25,
    alpha=0.75,
    legend=False,
    transform=PlateCarree(),
)
sns.scatterplot(
    data=data[data["cluster"] == -1],
    x="lon",
    y="lat",
    marker=".",
    color="w",
    edgecolor="k",
    legend=False,
    transform=PlateCarree(),
)
plt.title(f"Number of clusters: {n_clusters}")
plt.tight_layout()
plt.savefig("figs/esf/clusters.png", dpi=300)

# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection=Robinson())
ax.add_feature(LAND, color=".8")
ax.add_feature(OCEAN, color=".3")
ax.set_global()
sns.scatterplot(
    data=data[data["cluster"] != -1],
    x="lon",
    y="lat",
    ax=ax,
    marker=".",
    c="lightblue",
    edgecolor="None",
    label="data",
    transform=PlateCarree(),
)
sns.scatterplot(
    data=data[data["cluster"] == -1],
    x="lon",
    y="lat",
    ax=ax,
    label="noise",
    marker="x",
    color="r",
    transform=PlateCarree(),
)
sns.scatterplot(
    data=centroids,
    x="lonc",
    y="latc",
    ax=ax,
    label="centroids",
    s=40,
    edgecolor="w",
    transform=PlateCarree(),
)
ax.set_title(f"Number of clusters: {n_clusters}")
plt.tight_layout()
plt.savefig("figs/esf/cluster_centroids.png", dpi=300)

# %%
