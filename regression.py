# %%
import multiprocessing as mp

mp.set_start_method("forkserver")
# %%
from importlib import reload
import os
import pymc_models
from pprint import pprint


reload(pymc_models)


import xarray as xr
import pandas as pd
import numpy as np
import seaborn as sns
import pymc as pm
import arviz as az

from esda.moran import Moran
from libpysal.weights import full2W
import matplotlib.pyplot as plt

from utils.linalg import row_standardize


def gamma(zMC, npos):
    alpha = 0.1742
    numerator = 2.1480 - 6.1808 * (zMC + 0.6) ** alpha
    denominator = npos ** (0.1298) + 3.3534 / (zMC + 0.6) ** alpha
    return np.exp(numerator / denominator)


bandwidth = 16637

# %%
# Load data
# =============================================================================
data = pd.read_csv("data/esf/database_clustered.csv")
centroids = pd.read_csv("data/esf/centroids.csv")
C = pd.read_csv("data/esf/C.csv")
E_approx = pd.read_csv("data/esf/E_approx.csv")
CLplus = pd.read_csv("data/esf/CLplus.csv")
EL = pd.read_csv("data/esf/EL.csv")
E_mean_ = pd.read_csv("data/esf/E_mean_.csv")
E_std_ = pd.read_csv("data/esf/E_std_.csv")
eigvalsL = pd.read_csv("data/esf/eigvalsL.csv")

data_ext = pd.concat([data, E_approx], axis=1)

rads_centroids = np.deg2rad(centroids[["latc", "lonc"]].values)
n_clusters = centroids.shape[0]

X = data_ext[["d18Oc"]]
X["d18Oc_stdev"] = 0.1
X = pd.concat([X, data_ext[[f"E{i}" for i in range(E_approx.shape[1])]]], axis=1)
y = data_ext["sigmaT"]

models = {}
idata = {}


# %%
# Linear regression
# =============================================================================
sampler_config = {
    "draws": 4_000,
    "tune": 20_000,
    "chains": 4,
    "target_accept": 0.5,
    "cores": 1,
}
models["Linear"] = pymc_models.Linear(sampler_config=sampler_config)
idata["Linear"] = models["Linear"].fit(X, y)

# %%
# Polynomial regression 2 degree
# =============================================================================

sampler_config = {
    "draws": 4_000,
    "tune": 20_000,
    "chains": 4,
    "target_accept": 0.5,
    "cores": 1,
}
models["Poly2"] = pymc_models.Poly2(sampler_config=sampler_config)
idata["Poly2"] = models["Poly2"].fit(X, y)

# %%
# Linear ESF RHS regression
# =============================================================================


# RHS
n = X.shape[0]
npos = X.loc[:, "E0":].shape[1]
D = npos
zMC = 479  # pre-computed 479, 193
p0 = npos / (1 + gamma(zMC, npos))
data_sigma = 1.02  # precomputed 1.02, 0.78
tau0 = p0 / (D - p0) / np.sqrt(n)
s_tau = tau0 * data_sigma
sc = 0.5 * data_sigma / X.loc[:, "E0":].std(0)[0]
vc = 15
c2_alpha = vc / 2
c2_beta = vc * sc**2 / 2
tau_alpha = s_tau
print("p0:", p0)
print("tau0:", tau0)
print("s_tau:", s_tau)
print("sc:", sc)
print("c2_alpha:", c2_alpha)
print("c2_beta:", c2_beta)
# %%
model_config = {
    # GLM priors
    "a0_mu_prior": 0.0,
    "a1_mu_prior": 0.0,
    "a0_sigma_prior": 1000.0,
    "a1_sigma_prior": 1000.0,
    "sigma_y_alpha_prior": 1.0,
    "sigma_y_beta_prior": 1.0,
    # RHS prios
    "n_eigenvectors": npos,
    "c2_alpha": c2_alpha,
    "c2_beta": c2_beta,
    "tau_alpha": tau_alpha,
    "lbda_alpha": 1.0,
    "b_mu_prior": 0.0,
}
sampler_config = {
    "draws": 4_000,
    "tune": 20_000,
    "chains": 4,
    "target_accept": 0.5,
    "cores": 1,
}
models["LinearESF"] = pymc_models.LinearRHSESF(
    model_config=model_config, sampler_config=sampler_config
)
idata["LinearESF"] = models["LinearESF"].fit(X, y)


# %%
# Linear ESF Logit regression
# =============================================================================
model_config = {
    # GLM priors
    "a0_mu_prior": 0.0,
    "a1_mu_prior": 0.0,
    "a0_sigma_prior": 1000.0,
    "a1_sigma_prior": 1000.0,
    "sigma_y_alpha_prior": 1.0,
    "sigma_y_beta_prior": 1.0,
    # Logit prios
    "n_eigenvectors": npos,
    "tau": s_tau,
    "mu_lbda": 0.0,
    "sigma_lbda": 10.0,
    "bi_mu": 0.0,
}
sampler_config = {
    "draws": 4_000,
    "tune": 10_000,
    "chains": 4,
    "target_accept": 0.5,
    "cores": 1,
}
# models["LinearLogitESF"] = pymc_models.LinearLogitESF(
#     model_config=model_config, sampler_config=sampler_config
# )
# idata["LinearLogitESF"] = models["LinearLogitESF"].fit(X, y)

# %%
# Poly2 ESF RHS regression
# =============================================================================
n = X.shape[0]
npos = X.loc[:, "E0":].shape[1]
D = npos
zMC = 193  # pre-computed 479, 193
p0 = npos / (1 + gamma(zMC, npos))
data_sigma = 0.78  # precomputed 1.02, 0.78
tau0 = p0 / (D - p0) / np.sqrt(n)
s_tau = tau0 * data_sigma
sc = 0.5 * data_sigma / X.loc[:, "E0":].std(0)[0]
vc = 15
c2_alpha = vc / 2
c2_beta = vc * sc**2 / 2
tau_alpha = s_tau
print("p0:", p0)
print("tau0:", tau0)
print("s_tau:", s_tau)
print("c2_alpha:", c2_alpha)
print("c2_beta:", c2_beta)

model_config = {
    # GLM priors
    "a0_mu_prior": 0.0,
    "a1_mu_prior": 0.0,
    "a2_mu_prior": 0.0,
    "a0_sigma_prior": 1000.0,
    "a1_sigma_prior": 1000.0,
    "a2_sigma_prior": 1000.0,
    "sigma_y_alpha_prior": 1.0,
    "sigma_y_beta_prior": 1.0,
    # RHS prios
    "n_eigenvectors": npos,
    "c2_alpha": c2_alpha,
    "c2_beta": c2_beta,
    "tau_alpha": tau_alpha,
    "lbda_alpha": 1.0,
    "b_mu_prior": 0.0,
}
sampler_config = {
    "draws": 4_000,
    "tune": 20_000,
    "chains": 4,
    "target_accept": 0.5,
    "cores": 1,
}
models["Poly2ESF"] = pymc_models.Poly2RHSESF(
    model_config=model_config, sampler_config=sampler_config
)
idata["Poly2ESF"] = models["Poly2ESF"].fit(X, y)


# %%
# Poly2 Logit ESF regression
# =============================================================================
model_config = {
    # GLM priors
    "a0_mu_prior": 0.0,
    "a1_mu_prior": 0.0,
    "a2_mu_prior": 0.0,
    "a0_sigma_prior": 1000.0,
    "a1_sigma_prior": 1000.0,
    "a2_sigma_prior": 1000.0,
    "sigma_y_alpha_prior": 1.0,
    "sigma_y_beta_prior": 1.0,
    # Logit prios
    "n_eigenvectors": npos,
    "tau": s_tau,
    "mu_lbda": 0.0,
    "sigma_lbda": 10.0,
    "bi_mu": 0.0,
}
sampler_config = {
    "draws": 4_000,
    "tune": 10_000,
    "chains": 4,
    "target_accept": 0.9,
    "cores": 1,
}
# models["Poly2LogitESF"] = pymc_models.Poly2LogitESF(
#     model_config=model_config, sampler_config=sampler_config
# )
# idata["Poly2LogitESF"] = models["Poly2LogitESF"].fit(X, y)

# %%
for name, id in idata.items():
    try:
        with models[name].model:
            pm.compute_log_likelihood(id)
    except ValueError:
        pass


# %%
# Save models
# =============================================================================
print("Saving models...")
for name, model in models.items():
    os.makedirs(f"data/regression/{name}", exist_ok=True)
    model.save(f"data/regression/{name}/model.nc")
    az.to_netcdf(idata[name], f"data/regression/{name}/inference_data.nc")

# %%
# Load models
# =============================================================================
names = [
    "Linear",
    "Poly2",
    "LinearESF",
    "Poly2ESF",
    # "LinearLogitESF",
    # "Poly2LogitESF",
]
model_types = [
    pymc_models.Linear,
    pymc_models.Poly2,
    pymc_models.LinearRHSESF,
    pymc_models.Poly2RHSESF,
    # pymc_models.LinearLogitESF,
    # pymc_models.Poly2LogitESF,
]

for name, model in zip(names, model_types):
    if name not in models:
        try:
            models[name] = model.load(f"data/regression/{name}/model.nc")
        except FileNotFoundError:
            pass
    if name not in idata:
        try:
            idata[name] = az.from_netcdf(f"data/regression/{name}/inference_data.nc")
        except FileNotFoundError:
            pass


# %%
# Trace plots
# =============================================================================
print("Plotting trace plots...")
for model_name, id in idata.items():
    os.makedirs(f"figs/regression/{model_name}/", exist_ok=True)
    var_names = list(id.posterior.data_vars.keys())
    if "x_est" in var_names:
        var_names.remove("x_est")
    az.plot_trace(
        id,
        var_names=var_names,
    )
    plt.tight_layout()
    plt.savefig(f"figs/regression/trace_plot_{model_name}.png", dpi=200)
    plt.close()


# %%
# LOO & WAIC
# =============================================================================
print("Computing LOO & WAIC...")
loo = {}
waic = {}
for name, id in idata.items():
    loo[name] = az.loo(id)
    waic[name] = az.waic(id)

df_comp_loo = az.compare(loo)
df_comp_waic = az.compare(waic)

pprint(df_comp_loo)

# %%
fig = plt.figure(figsize=(14, 7))
axes = [fig.add_subplot(1, 2, i) for i in range(1, 3)]
az.plot_compare(df_comp_loo, ax=axes[0])
az.plot_compare(df_comp_waic, ax=axes[1])
axes[0].legend()
axes[1].legend()
fig.subplots_adjust(wspace=0.3, right=0.98)
plt.savefig("figs/regression/compare.png")
plt.close()


# %%
# Posterior predictive checks
# =============================================================================
print("Computing posterior predictive ...")
predictions = {}
for model_name, id in idata.items():
    preds = id.posterior_predictive["sigmaT"]
    preds.coords.update({"d18Oc": data["d18Oc"].values})
    predictions[model_name] = preds
    pred_means = preds.mean(("chain", "draw"))
    data_ext[f"sigmaT_pred_{model_name}"] = pred_means
    data_ext[f"sigmaT_res_{model_name}"] = data_ext["sigmaT"] - pred_means

data_ext.to_csv("data/regression/database_extended.csv", index=False)

# %%
# Morans I statistic
# =============================================================================
print("Computing Moran's I statistic...")


def compute_morans(data, W, col):
    W = full2W(W, ids=list(data.index.values))
    gm = Moran(data[col].values, W, permutations=0)
    return gm.I, gm.p_norm, gm.z_norm


W = row_standardize(C.values)
MCs = {}
for name, model in models.items():
    MCs[name] = compute_morans(data_ext, W, f"sigmaT_res_{name}")

# %%
# R2 score
# =============================================================================
from sklearn.metrics import r2_score

print("Computing R2 scores...")

r2_scores = {}
for name, model in models.items():
    r2_scores[name] = r2_score(data_ext["sigmaT"], data_ext[f"sigmaT_pred_{name}"])
# %%
from cartopy.crs import PlateCarree, Robinson
from cartopy.feature import OCEAN, LAND

cmap = "icefire"
fig = plt.figure(figsize=(14, 7))
axes = [fig.add_subplot(2, 2, i, projection=Robinson()) for i in range(1, 5)]
for ax, m in zip(axes, models.keys()):
    ax.add_feature(LAND, color=".8")
    ax.add_feature(OCEAN, color=".3")
    ax.set_global()
    data_ext.plot.scatter(
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
        "{:} | $\sigma_M =${:.4f} | $R^2$ = {:.3}".format(m, MCs[m][0], r2_scores[m])
    )
plt.tight_layout()
plt.savefig("figs/regression/prediction_residuals.png", dpi=300)


# %%
# Save models
# =============================================================================


data_ext_sorted = data_ext.sort_values("d18Oc")
fig = plt.figure(figsize=(14, 7))
axes = [fig.add_subplot(2, 2, i) for i in range(1, 5)]
for ax, m in zip(axes, models.keys()):
    data_ext_sorted.plot(
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
    data_ext_sorted.plot(x="d18Oc", y=f"sigmaT_pred_{m}", ax=ax, lw=0.5, legend=False)
    qs = (
        predictions[m]
        .quantile([0.025, 0.25, 0.75, 0.975], ("chain", "draw"))
        .sortby("d18Oc")
    )
    ax.fill_between(
        qs["d18Oc"],
        qs.sel(quantile=0.025),
        qs.sel(quantile=0.975),
        color="k",
        alpha=0.05,
    )
    ax.fill_between(
        qs["d18Oc"],
        qs.sel(quantile=0.25),
        qs.sel(quantile=0.75),
        color="k",
        alpha=0.1,
    )

    ax.set_title(m)
    ax.set_xlabel("$\delta^{18}O_c$ (‰ VSMOW)")
    ax.set_ylabel("$\sigma_T$ (kg m$^{-3}$)")
    sns.despine(ax=ax, trim=True, offset=5)

plt.tight_layout()
plt.savefig("figs/regression/prediction.png", dpi=300)


# %%
