import os

import numpy as np
import pandas as pd


countries = ["JP", "US", "CN", "IT", "ZA"]

def load_data(uuid: int, **kwargs: dict) -> np.ndarray:
    assert 1 <= uuid <= 2
    filepath = os.path.dirname(__file__)

    if uuid == 1:
        fn = "google"
    elif uuid == 2:
        fn = "owid"
        raise NotImplementedError("owid is not implemented yet.")

    if os.path.isfile(filepath + f"/{fn}-covid-data.csv.gz"):
        df = pd.read_csv(filepath + f"/{fn}-covid-data.csv.gz", index_col=0)
    else:
        print("Downloading data...")
        df = pd.read_csv("https://storage.googleapis.com/covid19-open-data/v3/epidemiology.csv")
        df.to_csv(filepath + f"/{fn}-covid-data.csv.gz")
    filtered_df = df[df["location_key"].isin(countries)][["date", "new_confirmed", "location_key"]]
    pivot_df = filtered_df.pivot(index='date', columns='location_key', values='new_confirmed')
    pivot_df.columns.name = None

    return pivot_df

# for OrbitMap
def load_arr_data(uuid: int, **kwargs: dict) -> np.ndarray:
    return load_data(uuid, **kwargs).to_numpy()
