import random

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LassoLarsIC, LinearRegression
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseModel
from src.models.modeplait.module.regime_storage import RegimeStorage

random.seed(0)
MAX_REGIMES = 30


class ModePlait(BaseModel):
    def __init__(self):
        super().__init__()

    def init_params(
        self,
        d: int,
        h: int,
        lcurr: int,
        lstep: int,
        lrprt: int,
        err_th: float,
        trunc_th: float,
        no_causal: bool,
        verbose: bool,
    ) -> None:
        self.d = d
        self.h = h
        self.lcurr = lcurr
        self.lstep = lstep
        self.lrprt = lrprt
        self.err_th = err_th
        self.trunc_th = trunc_th
        self.no_causal = no_causal
        self.verbose = verbose

    def initialize(self, X: np.ndarray) -> None:
        n, _ = X.shape
        self.regime_storage = RegimeStorage(
            trunc_th=self.trunc_th,
            err_th=self.err_th,
            no_causal=self.no_causal,
        )
        self.regime_storage.create_regime(
            Xc=X,
            h=self.h,
            append=True,
        )
        self.rgm_c = self.regime_storage[-1]
        self.S0 = None

    def estimate(self, Xc: np.ndarray) -> None:
        self.update_flag = True
        # find a best regime in regime set Θ.
        rgm_c, S0, err = self.regime_storage.get_best_regime(
            Xc=Xc,
            rgm_c_idx=self.rgm_c.idx,
        )
        if self.verbose:
            print(f"regime {rgm_c.idx} is selected")
            print(f"{err:.3f} vs {self.err_th} (err vs threshold)")
        if err > self.err_th and len(self.regime_storage) < MAX_REGIMES:
            # new regime is not necessary to update.
            self.update_flag = False
            # if candidate regime is not good, create new regime.
            new_rgm, new_S0, new_err = self.regime_storage.create_regime(
                Xc=Xc,
                h=self.h,
            )
            if new_err < self.err_th:
                # add new regime to regime set Θ only if it is good.
                if self.verbose:
                    print(f"regime {new_rgm.idx} is created")
                self.regime_storage.append(new_rgm)
                rgm_c = new_rgm
                S0 = new_S0
        # there are three types of regimes:
        # 1. good regime in regime set - update its parameters
        # 2. new good regime - add it to regime set
        # 3. new bad regime - do nothing, including update
        self.rgm_c = rgm_c
        self.S0 = S0

    def forecast(self, Xc: np.ndarray, support: bool = True) -> np.ndarray:
        Vf = np.real(
            self.rgm_c.predict(
                self.S0,
                Xc.shape[0] + self.lstep + self.lrprt,
                initial=True,
            )
        ) + np.mean(Xc, axis=0)
        if support:
            cond = self.__calc_cond(Xc, Vf)
            Vf[-(self.lstep + self.lrprt) :, cond] = Xc[-(self.lstep + self.lrprt) :, cond]
        return np.split(Vf[~np.isnan(Vf).any(axis=1)], [-(self.lstep + self.lrprt)])

    def update(self, Xc: np.ndarray) -> None:
        self.rgm_c.update(Xc, self.h)

    def get_causal_relationship(self, X: np.ndarray) -> np.ndarray:
        _, col_index = linear_sum_assignment(1 / np.abs(self.rgm_c.W))

        PW_ica = np.zeros_like(self.rgm_c.W)
        PW_ica[col_index] = self.rgm_c.W.copy()

        D = np.diag(PW_ica)[:, np.newaxis]

        W_estimate = PW_ica / D
        B_estimate = np.eye(len(W_estimate)) - W_estimate

        causal_order = self._estimate_causal_order(B_estimate)
        B_est = self._estimate_adjacency_matrix(X, causal_order)

        return np.where(np.abs(B_est) < 1.0, 0, B_est)

    def _estimate_causal_order(self, matrix):
        causal_order = None

        pos_list = np.argsort(np.abs(matrix), axis=None)
        pos_list = np.vstack(np.unravel_index(pos_list, matrix.shape)).T
        initial_zero_num = int(matrix.shape[0] * (matrix.shape[0] + 1) / 2)
        for i, j in pos_list[:initial_zero_num]:
            matrix[i, j] = 0

        for i, j in pos_list[initial_zero_num:]:
            causal_order = self._search_causal_order(matrix)
            if causal_order is not None:
                break
            else:
                matrix[i, j] = 0

        return causal_order

    def _search_causal_order(self, matrix):
        causal_order = []

        row_num = matrix.shape[0]
        original_index = np.arange(row_num)

        while 0 < len(matrix):
            row_index_list = np.where(np.sum(np.abs(matrix), axis=1) == 0)[0]
            if len(row_index_list) == 0:
                break

            target_index = row_index_list[0]

            causal_order.append(original_index[target_index])
            original_index = np.delete(original_index, target_index, axis=0)

            mask = np.delete(np.arange(len(matrix)), target_index, axis=0)
            matrix = matrix[mask][:, mask]

        if len(causal_order) != row_num:
            causal_order = None

        return causal_order

    def _estimate_adjacency_matrix(self, X, causal_order, prior_knowledge=None):
        if prior_knowledge is not None:
            pk = prior_knowledge.copy()
            np.fill_diagonal(pk, 0)

        B = np.zeros([X.shape[1], X.shape[1]], dtype="float64")
        for i in range(1, len(causal_order)):
            target = causal_order[i]
            predictors = causal_order[:i]

            if prior_knowledge is not None:
                predictors = [p for p in predictors if pk[target, p] != 0]

            if len(predictors) == 0:
                continue

            B[target, predictors] = self._predict_adaptive_lasso(X, predictors, target)

        return B

    def _predict_adaptive_lasso(self, X, predictors, target, gamma=1.0):
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        lr = LinearRegression()
        lr.fit(X_std[:, predictors], X_std[:, target])
        weight = np.power(np.abs(lr.coef_), gamma)
        reg = LassoLarsIC(criterion="bic")
        reg.fit(X_std[:, predictors] * weight, X_std[:, target])
        pruned_idx = np.abs(reg.coef_ * weight) > 0.0

        coef = np.zeros(reg.coef_.shape)
        if pruned_idx.sum() > 0:
            lr = LinearRegression()
            pred = np.array(predictors)
            lr.fit(X[:, pred[pruned_idx]], X[:, target])
            coef[pruned_idx] = lr.coef_

        return coef

    def __calc_cond(self, Xc: np.ndarray, Vf: np.ndarray):
        Xc_mean = np.mean(Xc, axis=0)
        Xc_std = np.std(Xc, axis=0)
        Xc_out = Xc[-1] - Xc_mean
        return np.logical_or(
            np.nanmax(Vf[self.h :] - Xc_mean, axis=0) > Xc_out + 3 * Xc_std,
            np.nanmin(Vf[self.h :] - Xc_mean, axis=0) < Xc_out - 3 * Xc_std,
        )
