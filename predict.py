# %%
from pymc_models import MODELS
import pandas as pd

import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from utils.kernels import bisquare_kernel
from utils.linalg import center_matrix, ones_vector
from utils.constants import AVG_EARTH_RADIUS

# %%
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
from utils.datasets import load_dataset
from models import predict_sigmaT

# %%
data_ext = pd.read_csv("data/regression/database_extended.csv")
is_within_lats = (data_ext.lat < 40) & (data_ext.lat > 20)
is_within_lons = (data_ext.lon > 55) & (data_ext.lon < 75)
is_within_area = is_within_lats & is_within_lons
data_as = data_ext[is_within_area]

# %%


md04 = load_dataset("rieger")
md04["lat"] = 25.0
md04["lon"] = 64.0

predictions = {}
for model in ["Linear", "Poly2", "LinearESF", "Poly2ESF"]:
    predictions[model] = predict_sigmaT(md04, model=model, name_d18Oc="d18Oc_mean")

# %%
df = xr.Dataset(predictions).isel(d18Oc=0).to_dataframe()
print(df.loc[:, "Linear":].std())
df = df.reset_index()
df = df.melt(
    id_vars=["chain", "draw", "d18Oc"], var_name=["model"], value_name="sigmaT"
)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
sns.kdeplot(
    ax=ax,
    data=df,
    x="sigmaT",
    hue="model",
    fill=True,
    common_norm=False,
    palette="tab20",
)
ax.scatter(data_as["sigmaT"], np.zeros_like(data_as["sigmaT"]), color="k")
ax.vlines(data_as["sigmaT"], 0, 0.5, color="k", alpha=0.5)
plt.xlabel("sigmaT [kg/m3]")
plt.ylabel("Probability density")
plt.title("Predicted sigmaT for one sample (185 yr BP) from the Arabian Sea")
plt.tight_layout()
plt.savefig("figs/regression/pred_samples.png", dpi=200)

# %%
