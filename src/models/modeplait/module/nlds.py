import lmfit
import numpy as np

from .regime import Regime

XTL = 1.0e-8
FTL = 1.0e-8
MAXFEV = 5000


class NLDS:
    def __init__(self, X: np.ndarray, regime: Regime) -> None:
        self.X = X
        self.d = regime.d
        self.r_ls = regime.r_ls
        H0 = (regime.W @ (X[: regime.h] - np.mean(X, axis=0)).copy().T).T
        self.S0 = [np.linalg.pinv(regime.Phis[j]) @ H0[:, j] for j in range(self.d)]
        self.re_S0 = [np.array(np.real(self.S0[i]), copy=True) for i in range(self.d)]
        self.im_S0 = [np.array(np.imag(self.S0[i]), copy=True) for i in range(self.d)]

    def fit_X0(self, regime: Regime) -> "NLDS":
        return nl_fit(self, regime)

    def generate(self, regime: Regime, n: int) -> np.ndarray:
        return regime.predict(self.S0, n) + np.mean(self.X, axis=0)


def nl_fit(nlds: "NLDS", regime: Regime) -> "NLDS":
    P = _createP(nlds)
    lmsol = lmfit.Minimizer(_objective, P, fcn_args=(nlds.X, nlds, regime))
    res = lmsol.leastsq(xtol=XTL, ftol=FTL, max_nfev=MAXFEV)
    nlds = _updateP(res.params, nlds)
    return nlds


def _createP(nlds: "NLDS") -> lmfit.parameter.Parameters:
    P = lmfit.Parameters()
    d = nlds.d
    r_ls = nlds.r_ls
    V = True
    for i in range(d):
        for j in range(r_ls[i]):
            P.add("re_S0_%i_%i" % (i, j), value=nlds.re_S0[i][j], vary=V)
            P.add("im_S0_%i_%i" % (i, j), value=nlds.im_S0[i][j], vary=V)
    return P


def _updateP(P: dict, nlds: "NLDS") -> "NLDS":
    d = nlds.d
    r_ls = nlds.r_ls
    for i in range(d):
        for j in range(r_ls[i]):
            nlds.re_S0[i][j] = P["re_S0_%i_%i" % (i, j)].value
            nlds.im_S0[i][j] = P["im_S0_%i_%i" % (i, j)].value
    nlds.S0 = [nlds.re_S0[i] + 1j * nlds.im_S0[i] for i in range(nlds.d)]
    return nlds


def _objective(P: dict, X: np.ndarray, nlds: "NLDS", regime: Regime) -> np.ndarray:
    n = X.shape[0]
    nlds = _updateP(P, nlds)
    pred = nlds.generate(regime, n)
    diff = np.abs(X.flatten() - pred.flatten())
    diff[np.isnan(diff)] = 0

    return diff
