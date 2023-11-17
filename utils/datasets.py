import pandas as pd


def load_dataset(name: str):
    """Load example datasets.

    Parameters
    ----------
    name : str
        Name of the dataset to load. Options are: 'weldeab', 'rieger', 'barathieu'.

    Returns
    -------
    pandas.DataFrame
        Example dataset.

    """
    path_example_data = "data/example/Data_test_salinity_caley.xlsx"
    sheets = {
        "weldeab": "Weldeab et al., 2022",
        "rieger": "Niclas Caley in prep",
        "barathieu": "Barathieu et al., in prep",
    }
    example_data = {}
    for k, sheet in sheets.items():
        ex_data = pd.read_excel(
            path_example_data,
            sheet_name=sheet,
            usecols="A:E",
            names=["age", "d18Oc_mean", "d18Oc_stdev", "T_mean", "T_stdev"],
        )
        ex_data["age"] = ex_data["age"].astype(int)
        ex_data = ex_data.set_index("age")
        ex_data["d18Oc_stdev"] = ex_data["d18Oc_stdev"] / 2
        ex_data["T_stdev"] = ex_data["T_stdev"] / 2
        example_data[k] = ex_data

    return example_data[name]
