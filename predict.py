# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
import xarray as xr
from sklearn.metrics.pairwise import haversine_distances

from models import predict_sigmaT
from pymc_models import MODELS
from utils.constants import AVG_EARTH_RADIUS
from utils.datasets import load_dataset
from utils.kernels import bisquare_kernel
from utils.linalg import center_matrix, ones_vector

# %%
data_ext = pd.read_csv("data/regression/full/database_extended.csv")
is_within_lats = (data_ext.lat < 40) & (data_ext.lat > 20)
is_within_lons = (data_ext.lon > 55) & (data_ext.lon < 75)
is_within_area = is_within_lats & is_within_lons
data_as = data_ext[is_within_area]

# %%


md04 = load_dataset("rieger")
md04["lat"] = 25.0
md04["lon"] = 64.0

# %%
from models import load_model, nystroem_approximation

X = md04.reset_index(inplace=False, drop=True)
predictors = X[["d18Oc_mean", "d18Oc_stdev"]]
latlons = X[["lat", "lon"]]
E_new = nystroem_approximation(latlons)
predictors = pd.concat([predictors, E_new], axis=1)

loaded_model = load_model("Poly2ESF")
pred = loaded_model.predict_posterior(predictors, random_seed=5)

# %%

predictions = {}
for model in ["Linear", "Poly2", "LinearESF", "Poly2ESF"]:
    predictions[model] = predict_sigmaT(md04, model=model, name_d18Oc="d18Oc_mean")

# %%
df = xr.Dataset(predictions).isel(d18Oc=0).to_dataframe()
print(df.loc[:, "Linear":].std())
df = df.reset_index(drop=True)
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
plt.savefig("figs/regression/full/pred_samples.png", dpi=200)

# %%
