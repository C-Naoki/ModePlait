import os
import pickle
import shutil
from datetime import datetime
from importlib import import_module
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from omegaconf import DictConfig, OmegaConf


class IOHelper:
    """
    Helper class for input/output operations.

    Attributes
    ----------
    input_dir : str
        Input directory path.
    out_dir : str
        Root output directory path.
    read_only : bool
        Read-only flag.
    """

    def __init__(self, io_cfg: DictConfig, read_only: bool = False) -> None:
        self.input_dir = io_cfg.input_dir
        self.out_dir = io_cfg.root_out_dir
        self.read_only = read_only

    def init_dir(self) -> None:
        assert self.read_only is False, "you set 'read_only=True'"
        assert hasattr(self, "out_dir"), "you didn't set 'out_dir'"
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)

    def mkdir(self, path: str = "", abs: str = False) -> None:
        assert hasattr(self, "out_dir"), "you didn't set 'out_dir'"
        path = path if abs else self.out_dir + path
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path)

    def load_data(self, io_cfg: DictConfig) -> pd.DataFrame:
        dataset_module = import_module(f"data.{self.input_dir}")
        kwargs = OmegaConf.to_container(io_cfg, resolve=True)
        if io_cfg.tag is None:
            data = dataset_module.load_data(**kwargs)
        elif io_cfg.tag == "arr":
            data = dataset_module.load_data(**kwargs)
            if io_cfg.input_dir == "synthetics":
                return data[0].values, data[1]
            else:
                return data.values
        else:
            raise NotImplementedError
        return data

    def savefig(self, fig: Figure, name: str = "") -> None:
        assert self.read_only is False, "you set 'read_only=True'"
        fig.savefig(self.out_dir + name)
        plt.close()

    def savepkl(self, obj: Any, name: str = "", abs: bool = False) -> None:
        assert self.read_only is False, "you set 'read_only=True'"
        if "." not in name:
            name += ".pkl"
        file_path = name if abs else self.out_dir + name
        f = open(file_path, "wb")
        pickle.dump(obj, f)
        f.close()

    def loadpkl(self, name: str = "", abs: bool = False) -> Any:
        if "." not in name:
            name += ".pkl"
        file_path = name if abs else self.out_dir + name
        try:
            f = open(file_path, "rb")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        obj = pickle.load(f)
        f.close()

        return obj


def backup() -> None:
    current_date = datetime.now()
    date_str = current_date.strftime("%Y%m%d")
    backup_path = f"backup/{date_str}/"
    if os.path.isdir(backup_path):
        shutil.rmtree(backup_path)
    shutil.copytree("out/", backup_path)


if __name__ == "__main__":
    backup()
