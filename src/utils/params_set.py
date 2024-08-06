import pandas as pd
from omegaconf import DictConfig, OmegaConf


def params_set(data: pd.DataFrame, cfg: DictConfig) -> DictConfig:
    """
    Set parameters to the configuration.

    Parameters
    ----------
    data: pd.DataFrame
        (Multivariate) Data Streams.
    cfg: DictConfig
        Configuration file.

    Returns
    -------
    DictConfig
        Updated Configuration file.

    Notes
    --------
    out_dir: str
        This output directory where the result of one experiment is saved.
        i.e., out_dir = out/mocap_1/window=1/name=modeplait/h=30/err_th=0.4/.../
    root_out_dir: str
        Thie root output directory where all results are saved.
        i.e., root_out_dir = out/
    """
    OmegaConf.set_struct(cfg, False)

    cfg.model.enc_in = data.shape[1]
    cfg.model.dec_in = data.shape[1]
    cfg.model.c_out = data.shape[1]
    cfg.model.pred_len = cfg.model.lstep + cfg.model.lrprt

    if cfg.io.input_dir == "synthetics":
        cfg.io.out_dir = (
            cfg.io.root_out_dir
            + f"{cfg.io.input_dir}_{cfg.io.uuid}/window={cfg.prep.moving_average}/"
            + f"n={cfg.io.n}/d={cfg.io.d}/p={cfg.io.p}/"
            + f"seq_type={cfg.io.seq_type}/noise_type={cfg.io.noise_type}/"
            + f"graph_type={cfg.io.graph_type}/seed={cfg.io.seed}/"
            + get_params_path(cfg.model)
        )
    else:
        cfg.io.out_dir = (
            cfg.io.root_out_dir
            + f"{cfg.io.input_dir}_{cfg.io.uuid}/window={cfg.prep.moving_average}/"
            + get_params_path(cfg.model)
        )

    return cfg


def get_params_path(cfg_model: DictConfig) -> str:
    params_path = "".join(f"{key}={value}/" for key, value in cfg_model.items())
    return params_path
