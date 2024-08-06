from typing import Optional, Tuple

import numpy as np

from src.utils.metrics import rmse

from .nlds import NLDS
from .regime import Regime


class RegimeStorage:
    def __init__(self, trunc_th: float, err_th: float, no_causal: bool = False) -> None:
        self.regimes = []
        self.update_params = []
        self.trunc_th = trunc_th
        self.err_th = err_th
        self.no_causal = no_causal

    def __call__(self) -> list[Regime]:
        return self.regimes

    def __getitem__(self, idx: int) -> Regime:
        return self.regimes[idx]

    def __len__(self) -> int:
        return len(self.regimes)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RegimeStorage):
            for rgm, o_rgm in zip(self, other):
                if rgm != o_rgm:
                    return False
            return True
        else:
            raise TypeError("Not supported type")
        return False

    def __iter__(self) -> "RegimeStorage":
        self._i = 0
        return self

    def __next__(self) -> Regime:
        if self._i == len(self):
            raise StopIteration
        regime = self.regimes[self._i]
        self._i += 1
        return regime

    def create_regime(
        self,
        Xc: np.ndarray,
        h: int,
        idx: Optional[int] = None,
        append: bool = False,
    ) -> Regime:
        if idx is None:
            idx = len(self)
        n, d = Xc.shape
        regime = Regime(idx=idx, no_causal=self.no_causal)
        regime.fit(Xc=Xc, h=h, trunc_th=self.trunc_th)
        try:
            nlds = NLDS(Xc, regime)
            nlds = nlds.fit_X0(regime)
            Vc = nlds.generate(regime, n)
            S0 = nlds.S0
        except ValueError:
            H0 = (regime.W @ (Xc[: regime.h] - np.mean(Xc, axis=0)).T).T
            S0 = [np.linalg.pinv(regime.Phis[j]) @ H0[:, j] for j in range(d)]
            Vc = regime.predict(S0, n) + np.mean(Xc, axis=0)
        err = rmse(Xc, Vc)
        if append:
            self.append(regime)

        return regime, S0, err

    def get_best_regime(
        self,
        Xc: np.ndarray,
        rgm_c_idx: int,
    ) -> Tuple[Regime, np.ndarray, float]:
        n, d = Xc.shape
        rgm_c = self[rgm_c_idx]
        H0 = (rgm_c.W @ (Xc[: rgm_c.h] - np.mean(Xc, axis=0)).T).T
        S0 = [np.linalg.pinv(rgm_c.Phis[j]) @ H0[:, j] for j in range(d)]
        min_Vc = rgm_c.predict(S0, n) + np.mean(Xc, axis=0)
        min_err = rmse(Xc, min_Vc)
        if min_err < self.err_th:
            return rgm_c, S0, min_err
        try:
            nlds = NLDS(Xc, rgm_c)
            nlds = nlds.fit_X0(rgm_c)
            min_Vc = nlds.generate(rgm_c, n)
            S0 = nlds.S0
        except ValueError:
            pass
        min_err = rmse(Xc, min_Vc)
        if min_err < self.err_th or len(self) == 1:
            return rgm_c, S0, min_err
        cand_rgm = rgm_c
        min_S0 = S0
        for regime in self:
            if regime == rgm_c:
                continue
            nlds = NLDS(Xc, regime)
            try:
                nlds = nlds.fit_X0(regime)
            except ValueError:
                continue
            Vc = nlds.generate(regime, n)
            err = rmse(Xc, Vc)
            if err < min_err:
                cand_rgm = regime
                min_err = err
                min_Vc = Vc
                min_S0 = nlds.S0
        return cand_rgm, min_S0, min_err

    def append(self, regime: Regime, update_param: Optional[dict] = None) -> None:
        self.regimes.append(regime)
