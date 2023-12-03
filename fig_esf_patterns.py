# %%
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from cartopy.crs import PlateCarree, Robinson
from cartopy.feature import LAND, OCEAN

# %%
EXPERIMENTS = ["full", "ruber"]
patterns = {}
for exp in EXPERIMENTS:
    data = pd.read_csv(f"data/esf/{exp}/database_clustered.csv")
    eigenvectors = pd.read_csv(f"data/esf/{exp}/E_approx.csv")
    patterns[exp] = pd.concat([data, eigenvectors], axis=1)

# %%

for exp, pats in patterns.items():
    col_names = pats.loc[:, "E0":].columns
    for name in col_names:
        vmax = abs(pats[name]).quantile(0.95).item()

        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(111, projection=Robinson())
        ax.add_feature(LAND, color=".8")
        ax.add_feature(OCEAN, color=".3")
        ax.set_global()
        pats.plot.scatter(
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
        path = f"figs/esf/patterns/{exp}/"
        os.makedirs(path, exist_ok=True)

        plt.savefig(path + f"{name}.png", dpi=200)
        plt.close()
