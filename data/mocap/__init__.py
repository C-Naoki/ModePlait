#  ---------------------------  #
# | No1: chicken_dance        | #
# | No2: exercise             | #
#  ---------------------------  #


import os

import numpy as np
import pandas as pd


def load_data(uuid: int, **kwargs: dict) -> np.ndarray:
    filepath = os.path.dirname(__file__)
    return pd.read_csv(filepath + f"/No{uuid}.csv.gz")


# for OrbitMap
def load_arr_data(uuid: int, **kwargs: dict) -> np.ndarray:
    filepath = os.path.dirname(__file__)

    data = pd.read_csv(filepath + f"/No{uuid}.csv.gz")

    return data.to_numpy()
