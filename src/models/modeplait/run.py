import time

import pandas as pd
from omegaconf import DictConfig

from src.models.base import BaseModel
from src.models.modeplait.mdb import MDB


def run(data: pd.DataFrame, model: BaseModel, cfg: DictConfig) -> None:
    # ------------------------------------ #
    #            preprocessing             #
    # ------------------------------------ #
    data = data.to_numpy()
    n, d = data.shape
    mdb = MDB(cfg, data)
    h = cfg.model.h
    lstep = cfg.model.lstep
    lcurr = cfg.model.lcurr
    lrprt = cfg.model.lrprt
    assert h < lcurr, "h must be less than lcurr"

    model.init_params(
        d=d,
        h=h,
        lcurr=lcurr,
        lstep=lstep,
        lrprt=lrprt,
        err_th=cfg.model.err_th,
        trunc_th=cfg.model.trunc_th,
        no_causal=cfg.model.no_causal,
        verbose=cfg.verbose,
    )

    # initializing
    model.initialize(data[:lcurr])

    for tm in range(lrprt, n, lrprt):
        tc = tm + lcurr
        tf = tm + lcurr + lstep
        te = tm + lcurr + lstep + lrprt
        if tc >= n:
            break
        if cfg.verbose:
            print(f"X[{tm}:{tc}] -> X[{tf}:{te}]")
        Xc = data[tm:tc]
        tic = time.monotonic()
        model.estimate(Xc)
        Vc, Ve = model.forecast(Xc)
        B = model.get_causal_relationship(Xc)

        if model.update_flag and not cfg.model.no_causal:
            model.update(Xc[-(lrprt + h) :])
        toc = time.monotonic()

        mdb.logging_results(
            tm=tm,
            rgm_idx=model.rgm_c.idx,
            time=toc - tic,
            rgm_c=model.rgm_c,
            Vc=Vc,
            Vf=Ve[-lrprt:],
            B_est=B,
        )
    print(mdb.summary)

    return {"MDB": mdb, "regime_storage": model.regime_storage}
