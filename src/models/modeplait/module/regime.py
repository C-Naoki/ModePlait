import concurrent.futures

import numpy as np
from sklearn.decomposition import fastica

from .dmd import DMD, delay_embedding

RANDOM_STATE = 42


class Regime:
    def __init__(self, idx: int, rho: float = 0.99, no_causal: bool = False) -> None:
        self.idx = idx
        self.rho = rho
        self.no_causal = no_causal

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Regime):
            return self.idx == other.idx
        else:
            raise TypeError("Not supported type")

    def get_ith_r(self, i: int) -> int:
        assert hasattr(self, "r_ls"), "please initialize first."
        return self.r_ls[i]

    def fit(self, Xc: np.ndarray, h: int, trunc_th: float) -> None:
        self.h = h
        self.n, self.d = Xc.shape
        self.I = np.eye(self.d)
        self.r_ls = [0] * self.d

        if not self.no_causal:
            K, W, E, self.mean = fastica(Xc, random_state=RANDOM_STATE, return_X_mean=True)
            self.W = W @ K
        else:
            self.mean = np.mean(Xc, axis=0)
            self.W = np.eye(self.d)
            E = Xc.copy()
        self.energies = np.sum((self.W @ (Xc - self.mean).T) ** 2, axis=1)

        Us = [0] * self.d
        Ps = [0] * self.d
        self.Phis = [0] * self.d
        self.Lambs = [0] * self.d
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.__parallel_fit, i, E, h, trunc_th)
                for i in range(self.d)
            ]
            for future in concurrent.futures.as_completed(futures):
                i, Phi, Lamb, U, P, r = future.result()
                self.Phis[i] = Phi
                self.Lambs[i] = Lamb
                Us[i] = U
                Ps[i] = P
                self.r_ls[i] = r

        self.update_param = {"U": Us, "P": Ps}

    def predict(self, S0: list, length: int, initial: bool = False) -> np.ndarray:
        # forecast X[self.h, length] if H0 is E[0, self.h]
        assert self.h <= length, "length must be larger than h."
        d = len(self.Phis)
        pred = np.zeros((d, length))
        pred[:, :] = np.nan
        for i in range(length - self.h + 1):
            for j in range(d):
                # independent components of each variable e_i(t)
                # note that this is projected onto the state space R^1 by g^(-1).
                pred[j, i + self.h - 1] = np.real(
                    self.Phis[j] @ np.diag(self.Lambs[j] ** i) @ S0[j]
                )[-1]
        if initial:
            for i in range(d):
                pred[i, : self.h] = np.real(self.Phis[i] @ S0[i])
        # consideration the effect of the other variables at time point t.
        return (np.linalg.inv(self.W) @ pred).T

    def update(self, X: np.ndarray, h: int) -> None:
        # Incremental ICA
        W = self.W.copy()
        for i in range(h, len(X)):
            new_x = X[i].copy()
            self.mean = (self.mean * self.n + new_x) / (self.n + 1)
            new_x = new_x - self.mean
            for j in range(self.d):
                new_e = W[j] @ new_x
                self.energies[j] = 0.96 * self.energies[j] + new_e**2
                err = new_x - new_e * W[j]
                W[j] += (new_e / self.energies[j]) * err
                new_x -= new_e * W[j]
            self.n += 1
        self.W = W

        # Windowed Incremental DMD
        E = (self.W @ X.T).T
        U, P = self.update_param.values()
        for i in range(self.d):
            Hi = delay_embedding(E[:, i].reshape(1, -1), h)
            for xt, yt in zip(Hi[:, :-1].T, Hi[:, 1:].T):
                Uih = self.__hermite(U[i])
                Phi_i = self.Phis[i]
                Lamb_i = self.Lambs[i]

                # projection xt onto k-dimensional subspace spanned by U
                xt_tilde = Uih @ xt
                yt_tilde = Uih @ yt

                # reconstruction A_tilde
                A_tilde = Uih @ Phi_i @ np.diag(Lamb_i) @ np.linalg.pinv(Phi_i) @ U[i]
                P[i] = P[i] / self.rho
                gamma = 1 / (1 + xt_tilde.T @ P[i] @ xt_tilde)

                # update A_tilde and P
                A_tilde += gamma * np.outer(yt_tilde - A_tilde @ xt_tilde, xt_tilde) @ P[i]
                P[i] = (P[i] - gamma * P[i] @ np.outer(xt_tilde, xt_tilde) @ P[i]) / self.rho
            self.Lambs[i], W = np.linalg.eig(A_tilde)
            self.Phis[i] = U[i] @ W

    def __hermite(self, X: np.ndarray) -> np.ndarray:
        return X.conj().T

    def __parallel_fit(self, i: int, E: np.ndarray, h: int, trunc_th: float):
        Hi = delay_embedding(E[None, :, i], h)
        dmd = DMD(trunc_th=trunc_th)
        Phi, Lamb, U, P = dmd.fit(Hi[:, :-1], Hi[:, 1:])
        r = len(Lamb)
        return i, Phi, Lamb, U, P, r
