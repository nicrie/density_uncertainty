# %%
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from cartopy.crs import PlateCarree, Robinson
from cartopy.feature import LAND, OCEAN

# %%
data = pd.read_csv("data/esf/database_clustered.csv")
patterns = pd.read_csv("data/esf/E_approx.csv")
data_ext = pd.concat([data, patterns], axis=1)
pattern_names = patterns.columns
# %%

for name in pattern_names:
    vmax = abs(data_ext[name]).quantile(0.95).item()

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111, projection=Robinson())
    ax.add_feature(LAND, color=".8")
    ax.add_feature(OCEAN, color=".3")
    ax.set_global()
    data_ext.plot.scatter(
        x="lon",
        y="lat",
        c=name,
        ec="k",
        alpha=0.75,
        ax=ax,
        cmap="icefire",
        transform=PlateCarree(),
        vmin=-vmax,
        vmax=vmax,
    )
    plt.tight_layout()
    ax.set_title(f"ESF {name}")
    fig.subplots_adjust(top=0.95)
    plt.savefig(f"figs/esf/patterns/{name}.png".format(), dpi=200)
    plt.close()

# %%
