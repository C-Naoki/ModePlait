#  ---------------------------  #
# | No1: beer_g_ma4           | #
#  ---------------------------  #


import os

import numpy as np
import pandas as pd


def load_data(uuid: int, **kwargs: dict) -> np.ndarray:
    filepath = os.path.dirname(__file__)

    data = pd.read_csv(filepath + f"/No{uuid}.csv.gz")
    data["date"] = pd.date_range(start="2004-1-1", periods=len(data), freq="W")
    data.set_index("date", inplace=True)

    return data


# for OrbitMap
def load_arr_data(uuid: int, **kwargs: dict) -> np.ndarray:
    filepath = os.path.dirname(__file__)

    data = pd.read_csv(filepath + f"/No{uuid}.csv.gz")

    return data.to_numpy()
