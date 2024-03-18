# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import haversine_distances
from utils.constants import AVG_EARTH_RADIUS
from utils.kernels import bisquare_kernel
from utils.linalg import center_matrix, ones_vector, row_standardize
from utils.metrics import centroid_on_sphere

# %%
# Load data
# =============================================================================
data = pd.read_csv("data/d18Oc_sigmaT_database.csv")
is_ruber = data["species"] == "G, ruber"
data_ruber = data[is_ruber]

n_samples = data.shape[0]
n_smaples_ruber = data_ruber.shape[0]


# %%
# Clustering
# =============================================================================
def assign_to_cluster(data, n_clusters):
    latlons = data[["lat", "lon"]].values
    rads = np.deg2rad(latlons)

    cl = HDBSCAN(
        metric="haversine",
        min_cluster_size=2,
        min_samples=1,
        cluster_selection_epsilon=0.01,
    )
    cl.fit(rads)
    return cl.labels_


def compute_centroids(data):
    # Compute centroids
    centroids = (
        data[["lon", "lat", "cluster"]]
        .groupby("cluster")
        .apply(lambda x: centroid_on_sphere(x.lon, x.lat))
    )
    if -1 in centroids.index:
        centroids.drop(-1, axis=0, inplace=True)

    centroids.columns = ["lon", "lat"]
    centroids = centroids[["lat", "lon"]]
    return centroids


data["cluster"] = assign_to_cluster(data, n_clusters=500)
centroids = compute_centroids(data)
n_clusters = centroids.shape[0]

data_ruber["cluster"] = assign_to_cluster(data_ruber, n_clusters=500)
centroids_ruber = compute_centroids(data_ruber)
n_clusters_ruber = centroids_ruber.shape[0]


ax = plt.axes()
data_ruber.plot.scatter(x="lon", y="lat", c=".8", marker=".", ax=ax)
centroids_ruber.plot.scatter(x="lon", y="lat", cmap="tab20", ax=ax)
plt.title(f"Number of clusters: {n_clusters}")


# %%
# Distance & kernel matrix
# =============================================================================
def esf_processing(data, centroids):
    n_samples = data.shape[0]
    n_clusters = centroids.shape[0]

    rads_data = np.deg2rad(data[["lat", "lon"]].values)
    rads_centroids = np.deg2rad(centroids[["lat", "lon"]].values)

    # Distance matrix
    dist = haversine_distances(rads_data, rads_data) * AVG_EARTH_RADIUS
    distL = haversine_distances(rads_centroids, rads_centroids) * AVG_EARTH_RADIUS
    distnL = haversine_distances(rads_data, rads_centroids) * AVG_EARTH_RADIUS

    bandwidth = dist.max(0).min()
    print("Bandwidth: {:.0f}".format(bandwidth))

    # Kernel matrix
    C = bisquare_kernel(dist, bandwidth)
    CL = bisquare_kernel(distL, bandwidth)
    CnL = bisquare_kernel(distnL, bandwidth)

    Cplus = C + np.eye(C.shape[0])
    CLplus = CL + np.eye(CL.shape[0])

    # Row-standardized weight matrix of centroids
    W = row_standardize(CL)

    # Eigendecomposition of matrix MCM
    eigvalsL, EL = np.linalg.eigh(
        center_matrix(n_clusters) @ CL @ center_matrix(n_clusters)
    )
    eigvalsL = eigvalsL[::-1]
    EL = EL[:, ::-1]
    is_positive = eigvalsL > 1e-5
    thres_cum_rel_fraction = 0.9999999
    n_eigvectors = (
        eigvalsL[is_positive].cumsum() / eigvalsL[is_positive].sum()
        <= thres_cum_rel_fraction
    ).sum()
    eigvalsL = eigvalsL[:n_eigvectors]
    EL = EL[:, :n_eigvectors]
    print("# samples:", n_samples)
    print("# clusters:", n_clusters)
    print("Shape of EL:", EL.shape)

    # Eigenfunction approximation using the Nyström extension
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
    E_std_ = E_approx.std(0)
    print("Mean E_std:", E_std_.round(2))
    print("Shape of E_approx:", E_approx.shape)

    # Prepare for saving
    cols = ["E{}".format(i) for i in range(n_clusters)]

    C = pd.DataFrame(C)
    CLplus = pd.DataFrame(CLplus, columns=cols)
    W = pd.DataFrame(W, columns=cols)
    EL = pd.DataFrame(EL, columns=cols[: len(eigvalsL)])
    E_approx = pd.DataFrame(E_approx, columns=cols[: len(eigvalsL)])
    eigvalsL = pd.Series(eigvalsL, name="eigvalsL", index=cols[: len(eigvalsL)])
    E_mean_ = pd.Series(E_mean_, name="E_mean_", index=cols[: len(eigvalsL)])
    E_std_ = pd.Series(E_std_, name="E_std_", index=cols[: len(eigvalsL)])
    return {
        "C": C,  # data kernel matrix
        "CLplus": CLplus,  # centroid kernel matrix (regularized)
        "W": W,  # row-standardized weight matrix (centroids)
        "EL": EL,  # eigenvectors of centroids
        "E_approx": E_approx,  # eigenvector approximation
        "eigvalsL": eigvalsL,  # eigenvalues of centroids
    }


esf_result = esf_processing(data, centroids)
esf_result_ruber = esf_processing(data_ruber, centroids_ruber)


# %%
data.to_csv("data/esf/full/database_clustered.csv", index=False)
centroids.to_csv("data/esf/full/centroids.csv", index=False)

data_ruber.to_csv("data/esf/ruber/database_clustered.csv", index=False)
centroids_ruber.to_csv("data/esf/ruber/centroids.csv", index=False)


for k, v in esf_result.items():
    v.to_csv("data/esf/full/{}.csv".format(k), index=False)

for k, v in esf_result_ruber.items():
    v.to_csv("data/esf/ruber/{}.csv".format(k), index=False)

# %%
