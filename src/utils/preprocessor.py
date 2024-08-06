from typing import Union

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn import preprocessing as prep
from tensorly.base import fold, unfold


def preprocessing(data: np.ndarray, prep_cfg: DictConfig) -> np.ndarray:
    if prep_cfg.moving_average > 1:
        data = moving_average(data, prep_cfg.moving_average)

    if prep_cfg.logarithm:
        data = log(data)

    if prep_cfg.zscore:
        data = scale(data)

    if prep_cfg.minmax_scale:
        data = minmax_scale(data)

    if prep_cfg.normalize:
        data = normalize(data)

    if prep_cfg.whitenoise > 0:
        data += np.random.normal(0, prep_cfg.whitenoise, size=data.shape)

    return data


def minmax_scale(data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    if isinstance(data, np.ndarray):
        if data.ndim > 2:
            scaled = prep.minmax_scale(unfold(data, 0))
            return fold(scaled, 0, data.shape)
        else:
            return prep.minmax_scale(data)

    elif isinstance(data, pd.DataFrame):
        numeric_col = data.select_dtypes(include=np.number).columns.tolist()
        data[numeric_col] = prep.minmax_scale(data[numeric_col].values)
        return data


def scale(data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    if isinstance(data, np.ndarray):
        if data.ndim > 2:
            scaled = prep.scale(unfold(data, 0))
            return fold(scaled, 0, data.shape)
        else:
            return prep.scale(data)

    if isinstance(data, pd.DataFrame):
        numeric_col = data.select_dtypes(include=np.number).columns.tolist()
        data[numeric_col] = prep.scale(data[numeric_col].values)
        return data


def normalize(X: np.ndarray) -> np.ndarray:
    if X.ndim > 2:
        scaled = prep.normalize(unfold(X, 0))
        return fold(scaled, 0, X.shape)
    else:
        return prep.normalize(X)


def log(X: np.ndarray) -> np.ndarray:
    if (X < 0).any():
        exit("data include negative values [log]")
    else:
        return np.log1p(X)


def moving_average(
    X: Union[np.ndarray, pd.DataFrame], window: int
) -> Union[np.ndarray, pd.DataFrame]:
    if isinstance(X, np.ndarray):
        n, d = X.shape
        _X = np.zeros((n - window + 1, d))
        for i in range(d):
            _X[:, i] = np.convolve(X[:, i], np.ones(window) / window, mode="valid")
        return _X
    elif isinstance(X, pd.DataFrame):
        return X.rolling(window).mean().dropna()
