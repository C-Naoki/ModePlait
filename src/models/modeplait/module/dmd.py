from typing import Optional, Tuple

import numpy as np
from numba import njit


@njit(cache=True)
def delay_embedding(data: np.ndarray, h: int) -> Tuple[np.ndarray, np.ndarray]:
    d, n = data.shape
    shaped_data = np.empty((h * d, n - h + 1), dtype=np.float64)
    for i in range(h, n + 1):
        shaped_data[:, i - h] = data[:, i - h : i].flatten()
    return shaped_data


class DMD:
    def __init__(self, trunc_th: float = 0.99, rho: float = 1.0) -> None:
        self.trunc_th = trunc_th
        self.rho = rho

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        w = np.array([self.rho**i for i in range(X.shape[1] - 1, -1, -1)])
        Xw = X * w
        Yw = Y * w
        U, S, V = self.__svd(Xw)
        Uh = self.__hermite(U)
        try:
            P_tilde = np.linalg.inv((Uh @ Xw) @ (Uh @ Xw).T)
        except np.linalg.LinAlgError:
            P_tilde = np.linalg.pinv((Uh @ Xw) @ (Uh @ Xw).T)
        A_tilde = Uh @ Yw @ V @ np.linalg.inv(S)
        Lamb, W = np.linalg.eig(A_tilde)
        Phi = U @ W

        return Phi, Lamb, U, P_tilde

    def __svd(
        self, X: np.ndarray, k: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        U, S, Vh = np.linalg.svd(X, full_matrices=False)
        if k is None:
            k = self.__get_k(S, X.shape)
        U = U[:, :k]
        S = np.diag(S[:k])
        V = self.__hermite(Vh[:k, :])

        return U, S, V

    def __get_k(self, S: np.ndarray, shape: tuple, min_dim: int = 2) -> int:
        if self.trunc_th:
            return max(int(np.argmax(np.cumsum(S) / np.sum(S) >= self.trunc_th) + 1), min_dim)
        else:
            return self.__svht(S, shape)

    def __hermite(self, X: np.ndarray) -> np.ndarray:
        return X.conj().T

    def __svht(self, S: np.ndarray, shape: tuple) -> int:
        beta = np.divide(*sorted(shape))
        omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
        tau = np.median(S) * omega
        rank = np.sum(S > tau)
        return rank
