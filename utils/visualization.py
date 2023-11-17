import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import xarray as xr


def plot_data(
    data: pd.DataFrame,
    name_mu: str,
    name_sigma: str,
    figname="",
    title: str = "",
    y_inverse: bool = False,
):
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.fill_between(
        data.index,
        data[name_mu] - 2 * data[name_sigma],
        data[name_mu] + 2 * data[name_sigma],
        color="k",
        alpha=0.1,
        label="95% CI",
    )
    ax.fill_between(
        data.index,
        data[name_mu] - 1 * data[name_sigma],
        data[name_mu] + 1 * data[name_sigma],
        color="k",
        alpha=0.2,
        label="68% CI",
    )
    ax.errorbar(
        data.index,
        data[name_mu],
        yerr=2 * data[name_sigma],
        ls="--",
        marker=".",
        color="darkred",
        lw=1,
        zorder=4,
        ecolor=".3",
        label="Original",
    )
    if y_inverse:
        ax.invert_yaxis()
    ax.set_title(title)
    ax.legend()
    sns.despine(ax=ax, trim=True, offset=5)

    plt.tight_layout()
    plt.savefig("figs/example/{:}.png".format(figname), dpi=300)
    plt.show()


def plot_data_samples(
    data: xr.DataArray, dim: str, figname: str = "", title: str = "", ylim=None
):
    qs = data.quantile([0.025, 0.25, 0.5, 0.75, 0.975], dim=dim)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.fill_between(
        qs["age"],
        qs.sel(quantile=0.025),
        qs.sel(quantile=0.975),
        color="k",
        alpha=0.1,
        label="95% CI",
    )
    ax.fill_between(
        qs["age"],
        qs.sel(quantile=0.25),
        qs.sel(quantile=0.75),
        color="k",
        alpha=0.2,
        label="50% CI",
    )
    qs.sel(quantile=0.5).plot(
        x="age", ax=ax, color="darkred", marker=".", ls="--", lw=1, label="Median"
    )
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_title(title)
    ax.legend()
    sns.despine(ax=ax, trim=True, offset=5)
    plt.tight_layout()
    plt.savefig("figs/example/{:}.png".format(figname), dpi=300)
    plt.show()


def plot_bootstrapped_salinity(bst_salinity, figname=""):
    fig = plt.figure(figsize=(8, 4))
    qs_sal = bst_salinity.quantile([0.025, 0.25, 0.5, 0.75, 0.975], "draw")
    ax = fig.add_subplot(111)
    qs_sal.sel(quantile=0.5).plot(
        x="age", ax=ax, color="darkred", marker=".", ls="--", lw=1, label="Median"
    )
    ax.fill_between(
        qs_sal["age"],
        qs_sal.sel(quantile=0.025),
        qs_sal.sel(quantile=0.975),
        color="darkred",
        alpha=0.2,
        label="95% CI",
    )
    ax.fill_between(
        qs_sal["age"],
        qs_sal.sel(quantile=0.25),
        qs_sal.sel(quantile=0.75),
        color="darkred",
        alpha=0.5,
        label="50% CI",
    )
    ax.set_xlabel("Age (kyr BP)")
    ax.set_ylabel("Salinity (psu)")
    ax.set_title("")
    ax.set_title("Bootstrapped salinity", loc="left")
    ax.legend()
    ax.set_ylim(25, 40)
    sns.despine(ax=ax, trim=True, offset=5)
    plt.tight_layout()
    plt.savefig("figs/example_salinity_uncertainty_{:}.png".format(figname), dpi=300)
    plt.show()


def plot_bootstrapped_salinity_local(salinity, figname=""):
    qs_sal_local = salinity.quantile([0.025, 0.25, 0.5, 0.75, 0.975], "draw")
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    qs_sal_local.sel(quantile=0.5).plot(
        x="age", ax=ax, color="darkred", marker=".", ls="--", lw=1, label="Median"
    )
    ax.fill_between(
        qs_sal_local["age"],
        qs_sal_local.sel(quantile=0.025),
        qs_sal_local.sel(quantile=0.975),
        color="darkred",
        alpha=0.2,
        label="95% CI",
    )
    ax.fill_between(
        qs_sal_local["age"],
        qs_sal_local.sel(quantile=0.25),
        qs_sal_local.sel(quantile=0.75),
        color="darkred",
        alpha=0.5,
        label="50% CI",
    )
    ax.set_xlabel("Age (kyr BP)")
    ax.set_ylabel("Salinity (psu)")
    ax.set_title("")
    ax.set_title("Bootstrapped salinity (local)", loc="left")
    ax.set_ylim(25, 40)
    ax.legend()
    sns.despine(ax=ax, trim=True, offset=5)
    plt.tight_layout()
    plt.savefig(f"figs/example_salinity_uncertainty_local_{figname}.png", dpi=300)
    plt.show()
