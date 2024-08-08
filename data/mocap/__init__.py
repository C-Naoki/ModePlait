#  ---------------------------  #
# | No1: chicken_dance        | #
# | No3: exercise             | #
#  ---------------------------  #


import os

import numpy as np
import pandas as pd


def load_data(uuid: int, **kwargs: dict) -> np.ndarray:
    filepath = os.path.dirname(__file__)

    data = pd.read_csv(filepath + f"/No{uuid}.csv.gz")

    if "date" not in data.columns:
        data["date"] = pd.date_range(start='2021-01-01', periods=len(data), freq='D')
        data.set_index("date", inplace=True)

    return data

# for OrbitMap
def load_arr_data(uuid: int, **kwargs: dict) -> np.ndarray:
    filepath = os.path.dirname(__file__)

    data = pd.read_csv(filepath + f"/No{uuid}.csv.gz")

    return data.to_numpy()
