# %%
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import haversine_distances

from utils.metrics import centroid_on_sphere
from utils.kernels import bisquare_kernel
from utils.linalg import center_matrix, ones_vector, row_standardize
from utils.constants import AVG_EARTH_RADIUS

# %%
# Load data
# =============================================================================
data = pd.read_csv("data/d18Oc_sigmaT_database.csv")

n_samples = data.shape[0]

# %%
# Clustering
# =============================================================================
latlons = data[["lat", "lon"]].values
rads = np.deg2rad(latlons)

cl = HDBSCAN(
    metric="haversine",
    min_cluster_size=2,
    min_samples=1,
    # cluster_selection_epsilon=0.01,
)
cl.fit(rads)
# cl = AgglomerativeClustering(
#     n_clusters=50,
#     metric="precomputed",
#     linkage="average",
# )
# dmat = haversine_distances(rads, rads) * AVG_EARTH_RADIUS
# cl.fit(dmat)
data["cluster"] = cl.labels_

# Compute centroids
centroids = (
    data[["lon", "lat", "cluster"]]
    .groupby("cluster")
    .apply(lambda x: centroid_on_sphere(x.lon, x.lat))
)
centroids.columns = ["lonc", "latc"]
if -1 in centroids.index:
    centroids.drop(-1, axis=0, inplace=True)

centroids = centroids[["latc", "lonc"]]
n_clusters = centroids.shape[0]

import matplotlib.pyplot as plt

ax = plt.axes()
data.plot.scatter(x="lon", y="lat", c=".8", marker=".", ax=ax)
centroids.plot.scatter(x="lonc", y="latc", cmap="tab20", ax=ax)
plt.title(f"Number of clusters: {n_clusters}")


# Extend and save data
data.to_csv("data/esf/database_clustered.csv", index=False)
centroids.to_csv("data/esf/centroids.csv", index=False)

# %%
# Distance matrix
# =============================================================================
rads_centroids = np.deg2rad(centroids[["latc", "lonc"]].values)
dist = haversine_distances(rads, rads) * AVG_EARTH_RADIUS
distL = haversine_distances(rads_centroids, rads_centroids) * AVG_EARTH_RADIUS
distnL = haversine_distances(rads, rads_centroids) * AVG_EARTH_RADIUS

bandwidth = dist.max(0).min()
print("Bandwidth: {:.0f}".format(bandwidth))

# %%
# Kernel matrix
# =============================================================================
C = bisquare_kernel(dist, bandwidth)
CL = bisquare_kernel(distL, bandwidth)
CnL = bisquare_kernel(distnL, bandwidth)

Cplus = C + np.eye(C.shape[0])
CLplus = CL + np.eye(CL.shape[0])

# Row-standardized weight matrix of centroids
W = row_standardize(CL)

# %%
# Eigendecompose matrix MCM
# =============================================================================
eigvalsL, EL = np.linalg.eigh(
    center_matrix(n_clusters) @ CL @ center_matrix(n_clusters)
)
eigvalsL = eigvalsL[::-1]
EL = EL[:, ::-1]
is_positive = eigvalsL > 0
thres_cum_rel_fraction = 0.9999
n_eigvectors = (
    eigvalsL[is_positive].cumsum() / eigvalsL[is_positive].sum()
    <= thres_cum_rel_fraction
).sum()
eigvalsL = eigvalsL[:n_eigvectors]
EL = EL[:, :n_eigvectors]
print("# samples:", n_samples)
print("# clusters:", n_clusters)
print("Shape of EL:", EL.shape)

# %%
# Eigenfunction approximation using the Nyström extension
# =============================================================================
E_approx = (
    (
        CnL
        - np.kron(
            ones_vector(n_samples), ones_vector(n_clusters).T @ CLplus / n_clusters
        )
    )
    @ EL
    @ np.diag(1.0 / (eigvalsL))
)
E_mean_ = E_approx.mean(0)
# E_approx -= E_mean_
E_std_ = E_approx.std(0)
print("Mean E_std:", E_std_.round(2))
# E_approx /= E_std_
print("Shape of E_approx:", E_approx.shape)

# %%
cols = ["E{}".format(i) for i in range(n_clusters)]

C = pd.DataFrame(C)
CLplus = pd.DataFrame(CLplus, columns=cols)
W = pd.DataFrame(W, columns=cols)
EL = pd.DataFrame(EL, columns=cols[: len(eigvalsL)])
E_approx = pd.DataFrame(E_approx, columns=cols[: len(eigvalsL)])
eigvalsL = pd.Series(eigvalsL, name="eigvalsL", index=cols[: len(eigvalsL)])
E_mean_ = pd.Series(E_mean_, name="E_mean_", index=cols[: len(eigvalsL)])
E_std_ = pd.Series(E_std_, name="E_std_", index=cols[: len(eigvalsL)])

# %%

C.to_csv("data/esf/C.csv", index=False)  # Kernel matrix (data)
CLplus.to_csv(
    "data/esf/CLplus.csv", index=False
)  # Kernel matrix (centroids, regularized)
W.to_csv("data/esf/W.csv", index=False)  # Row-standardized weight matrix (centroids)
EL.to_csv("data/esf/EL.csv", index=False)  # Eigenvectors of centroids
E_approx.to_csv("data/esf/E_approx.csv", index=False)  # Eigenfunction approximation
eigvalsL.to_csv("data/esf/eigvalsL.csv", index=True)  # Eigenvalues of centroids
E_mean_.to_csv(
    "data/esf/E_mean_.csv", index=True
)  # Mean of eigenfunction approximation
E_std_.to_csv(
    "data/esf/E_std_.csv", index=True
)  # Standard deviation of eigenfunction approximation

# %%
