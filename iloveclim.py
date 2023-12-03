# %%
import pandas as pd


# %%
dataPI = pd.read_csv(
    "data/iloveclim/iLOVECLIMX2Y2PI.txt",
    sep="\t",
    header=0,
    skiprows=0,
    names=["d18Oc", "sigmaT"],
)
dataLGM = pd.read_csv(
    "data/iloveclim/iLOVECLIMX2Y2LGM.txt",
    sep="\t",
    header=0,
    skiprows=0,
    names=["d18Oc", "sigmaT"],
)

is_valid_PI = (dataPI != 0).any(axis=1)
is_valid_LGM = (dataLGM != 0).any(axis=1)

dataPI = dataPI[is_valid_PI]
dataLGM = dataLGM[is_valid_LGM]

dataLGM = dataLGM[dataLGM < 30]
dataLGM = dataLGM[~dataLGM.isnull().any(axis=1)]
# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.set_xlim(-6, 4)
ax1.set_ylim(18, 30)
ax2 = fig.add_subplot(212, sharex=ax1, sharey=ax1)
dataPI.plot.scatter(x="d18Oc", y="sigmaT", marker=".", ax=ax1, s=5, alpha=0.3)
dataLGM.plot.scatter(x="d18Oc", y="sigmaT", marker=".", ax=ax2, s=5, alpha=0.3)

ax1.set_ylabel(r"$\sigma_{T}$ [$kg\cdot m^{-3}$]")
ax2.set_ylabel(r"$\sigma_{T}$ [$kg\cdot m^{-3}$]")
ax2.set_xlabel(r"$\delta^{18}O_{c}$ [‰]")

ax1.set_title("PI")
ax2.set_title("LGM")
plt.tight_layout()
plt.savefig("figs/iloveclim/scatterplot.png", dpi=300)
# %%

import pymc_models


import pandas as pd
import numpy as np
import seaborn as sns
import arviz as az


models = {}
idata = {}


# %%
# Polynomial regression 2 degree (PI)
# =============================================================================
X = dataPI[["d18Oc"]]
X["d18Oc_stdev"] = 0.1
y = dataPI["sigmaT"]

sampler_config = {
    "draws": 1_000,
    "tune": 1_000,
    "chains": 4,
    "target_accept": 0.5,
    "cores": 1,
}
models["PI"] = pymc_models.Poly2(sampler_config=sampler_config)
idata["PI"] = models["PI"].fit(X, y)

# %%
# Polynomial regression 2 degree (LGM)
# =============================================================================
X = dataLGM[["d18Oc"]]
X["d18Oc_stdev"] = 0.1
y = dataLGM["sigmaT"]

sampler_config = {
    "draws": 1_000,
    "tune": 1_000,
    "chains": 4,
    "target_accept": 0.5,
    "cores": 1,
}
models["LGM"] = pymc_models.Poly2(sampler_config=sampler_config)
idata["LGM"] = models["LGM"].fit(X, y)

# %%
az.plot_trace(idata["PI"], var_names=["a0", "a1", "a2", "sigma_y"])
plt.savefig("figs/iloveclim/traceplot_PI.png", dpi=300)
# %%
df = az.summary(idata["PI"], var_names=["a0", "a1", "a2", "sigma_y"])
df.to_csv("data/iloveclim/summary_PI.csv")
