import os
import pickle
import shutil
from typing import Optional, Tuple

import imageio
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

from src.models.modeplait.module.regime import Regime
from src.utils.metrics import mae, rmse


class MDB:
    """
    Model database for storing simulation results (i.e., parameters).

    Parameters
    ----------
    cfg : DictConfig
        Configuration file.
    X : np.ndarray
        Data matrix.

    Notes
    --------
    if you want to save additive parameters, you follow the below steps.
    1. Add a new key to the `results` dictionary in the `__init__`.
    2. Fix the `logging_results` function to save the new parameter.
    3. Make a new private logging function (i.e., __logging_time)
    4. Add a new property to access the new parameter.
    """
    width = 12
    plot_style = {
        "color": "lightgray",
        "alpha": 0.6,
        "linewidth": 5,
    }

    def __init__(self, cfg: DictConfig, X: np.ndarray) -> None:
        self.n, self.d = X.shape
        self.out_dir = cfg.io.out_dir
        self.train_size = self.n // 3
        self.X = X
        self.metadata = {
            "lcurr": cfg.model.lcurr,
            "lstep": cfg.model.lstep,
            "lrprt": cfg.model.lrprt,
            "err_th": cfg.model.err_th,
            "trunc_th": cfg.model.trunc_th,
            "h": cfg.model.h,
        }
        self.results = {
            "time": np.nan * np.zeros(self.n),
            "regime": [None] * self.n,
            "rgm_idx": np.zeros(self.n),
            "B_est": [None] * self.n,
        }
        for d_i in range(self.d):
            self.results[f"Vc{d_i}"] = np.nan * np.zeros(self.n)
            self.results[f"Vf{d_i}"] = np.nan * np.zeros(self.n)

    def logging_results(
        self,
        tm: int,
        rgm_idx: int,
        time: float,
        rgm_c: np.ndarray,
        Vc: np.ndarray,
        Vf: np.ndarray,
        B_est: np.ndarray,
    ) -> None:
        self.__logging_time(tm, time)
        self.__logging_rgm(tm, rgm_c)
        self.__logging_rgm_idx(tm, rgm_idx)
        self.__logging_Vc(tm, Vc)
        self.__logging_Vf(tm, Vf)
        self.__logging_B_est(tm, B_est)

    def plot_time(self) -> None:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_yscale("log")
        ax.set_xlabel("Sequence Length", fontsize=20)
        ax.set_ylabel("Wall Clock Time (s)", fontsize=20)
        ax.set_xlim(self.train_size, self.n - 20)
        ax.plot(self.time, "x:", label="ModePlait")

        fig.tight_layout()
        fig.align_labels()
        with open(self.out_dir + "time.pkl", "wb") as f:
            pickle.dump(fig, f)
        plt.clf()
        plt.close()

    def plot_Vc(self, fmt: str = "png") -> None:
        fig = plt.figure(figsize=(12, 4.5))
        gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2.5, 1])
        axes = {"seq": fig.add_subplot(gs[0]), "rgm_idx": fig.add_subplot(gs[1])}

        n_rgm = len(np.unique(self.rgm_idx))
        err = rmse(self.X[self.train_size :], self.Vc[self.train_size :])

        axes["seq"].set_title(f"Total Fitting Error: {err:.3f}")
        axes["seq"].set_ylabel("Value")
        axes["seq"].set_xlim(0, self.n)
        axes["seq"].set_ylim(self.X.min() * 1.1, self.X.max() * 1.1)
        axes["seq"].tick_params(labelbottom=False, bottom=False)
        axes["seq"].text(
            x=self.train_size / 2,
            y=self.X.max(),
            s="Train",
            fontsize=16,
            color="green",
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="center",
        )
        axes["seq"].text(
            x=(self.n + self.train_size) / 2,
            y=self.X.max(),
            s="Test",
            fontsize=16,
            color="orangered",
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="center",
        )
        axes["seq"].axvspan(0, self.train_size, color="green", alpha=0.1)
        axes["seq"].axvspan(self.train_size, self.n, color="orange", alpha=0.1)
        axes["seq"].plot(self.X, **self.plot_style)
        axes["seq"].plot(self.Vc)
        axes["rgm_idx"].set_xlabel("Time")
        axes["rgm_idx"].set_ylabel("Regime")
        axes["rgm_idx"].set_xlim(0, self.n)
        axes["rgm_idx"].set_ylim(0, n_rgm)
        axes["rgm_idx"].set_yticks([i + 0.5 for i in range(n_rgm)])
        axes["rgm_idx"].set_yticklabels([f"#{i+1}" for i in range(n_rgm)])
        axes["rgm_idx"].hlines(
            y=list(range(n_rgm + 1)),
            xmin=0,
            xmax=self.n,
            color="gray",
            alpha=0.6,
        )
        for i in range(self.n):
            axes["rgm_idx"].axvspan(
                xmin=i,
                xmax=i + 1,
                ymin=self.rgm_idx[i] / n_rgm,
                ymax=(self.rgm_idx[i] + 1) / n_rgm,
            )
        fig.tight_layout()
        fig.align_labels()
        fig.savefig(self.out_dir + f"Vc.{fmt}")
        with open(self.out_dir + "Vc.pkl", "wb") as f:
            pickle.dump(fig, f)
        plt.clf()
        plt.close()

    def plot_Vf(self, fmt: str = "png") -> None:
        fig = plt.figure(figsize=(12, 4.5))
        gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2.5, 1])
        axes = {"seq": fig.add_subplot(gs[0]), "rgm_idx": fig.add_subplot(gs[1])}

        n_rgm = len(np.unique(self.rgm_idx))
        err = rmse(self.X[self.train_size :], self.Vf[self.train_size :])

        axes["seq"].set_title(f"Total Forecast Error: {err:.3f}")
        axes["seq"].set_ylabel("Value")
        axes["seq"].set_xlim(0, self.n)
        axes["seq"].set_ylim(self.X.min() * 1.1, self.X.max() * 1.1)
        axes["seq"].tick_params(labelbottom=False, bottom=False)
        axes["seq"].text(
            x=self.train_size / 2,
            y=self.X.max(),
            s="Train",
            fontsize=16,
            color="green",
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="center",
        )
        axes["seq"].text(
            x=(self.n + self.train_size) / 2,
            y=self.X.max(),
            s="Test",
            fontsize=16,
            color="orangered",
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="center",
        )
        axes["seq"].axvspan(0, self.train_size, color="green", alpha=0.1)
        axes["seq"].axvspan(self.train_size, self.n, color="orange", alpha=0.1)
        axes["seq"].plot(self.X, **self.plot_style)
        axes["seq"].plot(self.Vf)
        axes["rgm_idx"].set_xlabel("Time")
        axes["rgm_idx"].set_ylabel("Regime")
        axes["rgm_idx"].set_xlim(0, self.n)
        axes["rgm_idx"].set_ylim(0, n_rgm)
        axes["rgm_idx"].set_yticks([i + 0.5 for i in range(n_rgm)])
        axes["rgm_idx"].set_yticklabels([f"#{i+1}" for i in range(n_rgm)])
        axes["rgm_idx"].hlines(
            y=list(range(n_rgm + 1)),
            xmin=0,
            xmax=self.n,
            color="gray",
            alpha=0.6,
        )
        for i in range(self.n):
            axes["rgm_idx"].axvspan(
                xmin=i,
                xmax=i + 1,
                ymin=self.rgm_idx[i] / n_rgm,
                ymax=(self.rgm_idx[i] + 1) / n_rgm,
            )
        fig.tight_layout()
        fig.align_labels()
        fig.savefig(self.out_dir + f"Vf.{fmt}")
        with open(self.out_dir + "Vf.pkl", "wb") as f:
            pickle.dump(fig, f)
        plt.clf()
        plt.close()

    def plot_latent(
        self,
        tm: int,
        Ec: np.ndarray,
        fmt: str = "png",
        labels: Optional[list] = None,
        dates: Optional[list] = None,
    ) -> None:
        if not os.path.isdir(self.out_dir + "/latent"):
            os.mkdir(self.out_dir + "/latent")
            os.mkdir(self.out_dir + "/latent/pkl")
            os.mkdir(self.out_dir + "/latent/images")
        tc = tm + self.lcurr
        te = tm + self.lcurr + self.lstep + self.lrprt
        if te >= self.n:
            return
        xmin, xmax = tm - 0.05 * (tc - tm), tc + 0.05 * (tc - tm)
        # ymin, ymax = -4, 4
        ymin, ymax = Ec.min() * 1.1, Ec.max() * 1.1
        if not labels:
            labels = [f"x{i}" for i in range(self.d)]

        fig, ax = plt.subplots(figsize=(4, 2))
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_xticks(list(range((tm // 15) * 15, tc, 15)))
        ax.set_yticks([-3, -1, 1, 3])
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        for i in range(self.d):
            ax.plot(np.arange(tm, tc), Ec[:, i], linewidth=2.5, label=labels[i])

        ax.set_xticklabels(dates[list(range((tm // 15) * 15, tc, 15))])
        fig.tight_layout()
        fig.align_labels()
        fig.savefig(self.out_dir + f"latent/images/tm={tm}.{fmt}")
        with open(self.out_dir + f"latent/pkl/tm={tm}.pkl", "wb") as f:
            pickle.dump((fig, ax), f)
        plt.clf()
        plt.close()

    def plot_snapshot(
        self,
        tm: int,
        Ve: np.ndarray,
        fmt: str = "png",
        labels: Optional[list] = None,
        dates: Optional[list] = None,
    ) -> None:
        if not os.path.isdir(self.out_dir + "/snapshots"):
            os.mkdir(self.out_dir + "/snapshots")
            os.mkdir(self.out_dir + "/snapshots/pkl")
            os.mkdir(self.out_dir + "/snapshots/images")
        tc = tm + self.lcurr
        tf = tm + self.lcurr + self.lstep
        te = tm + self.lcurr + self.lstep + self.lrprt
        tm = tm + self.h - 1
        if te >= self.n:
            return
        xmin, xmax = tm - 0.05 * (te - tm), te + 0.05 * (te - tm)
        ymin, ymax = min(self.X[tm:te].min() * 1.1, -1.1), max(self.X[tm:te].max() * 1.1, 1.1)
        if not labels:
            labels = [f"x{i}" for i in range(self.d)]

        fig, ax = plt.subplots(figsize=(4, 2))
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_xticks(list(range((tm // 20) * 20, te, 20)))
        if ymax > 5:
            ax.set_yticks([-4, -1, 2, 5])
        else:
            ax.set_yticks([-3, -1, 1, 3])
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax.plot(self.X, **self.plot_style)
        for i in range(self.d):
            ax.plot(
                np.arange(tm, te),
                np.concatenate((self.Vc[tm:tc], Ve))[:, i],
                linewidth=2.5,
                label=labels[i],
            )
        ax.vlines(tm, ymin, ymax, color="blue")
        ax.vlines(tc, ymin, ymax, color="blue")
        ax.vlines(tf, ymin, ymax, color="red")
        if self.lrprt > 1:
            ax.vlines(te, ymin, ymax, color="red")

        # fig.legend(bbox_to_anchor=(0.95, 0.95), loc='upper right')
        ax.set_xticklabels(dates[list(range((tm // 20) * 20, te, 20))])
        ax.xaxis.set_tick_params(rotation=0)
        fig.tight_layout()
        fig.align_labels()
        fig.savefig(self.out_dir + f"snapshots/images/tm={tm - self.h + 1}.{fmt}")
        with open(self.out_dir + f"snapshots/pkl/tm={tm - self.h + 1}.pkl", "wb") as f:
            pickle.dump((fig, ax), f)
        plt.clf()
        plt.close()

    def plot_modes(self, tm: int, fmt: Optional[str] = None) -> None:
        if not os.path.isdir(self.out_dir + "/modes"):
            os.mkdir(self.out_dir + "/modes")
            os.mkdir(self.out_dir + "/modes/pkl")
            if fmt:
                os.mkdir(self.out_dir + "/modes/images")
        fig = plt.figure(figsize=(self.d * 3, 3))
        axes = [fig.add_subplot(1, self.d, i + 1) for i in range(self.d)]
        regime = self.regime[tm]
        COLORS = [i["color"] for i in plt.rcParams["axes.prop_cycle"]]

        def circle(radius: float = 1.0) -> Tuple[list, list]:
            x, y = [], []
            for angle in np.linspace(-180, 180, 360):
                x.append(radius * np.sin(np.radians(angle)))
                y.append(radius * np.cos(np.radians(angle)))
            return x, y

        fig.suptitle(f"Regime #{str(regime.idx)}", fontsize=20)
        for i in range(self.d):
            ax = axes[i]
            ax.scatter([0], [0], c="black", s=5)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.plot(*circle(), c="gray", linestyle="dashed", zorder=1)
            ax.set_title(f"AVG(λ) = {np.mean(np.abs(regime.Lambs[i])) - 1:.3f}")
            for lamb in regime.Lambs[i]:
                ax.scatter(
                    np.real(lamb),
                    np.imag(lamb),
                    c=COLORS[i],
                    s=50,
                    zorder=2,
                )
                ax.set_aspect("equal")
        fig.tight_layout()
        fig.align_labels()
        if fmt:
            fig.savefig(self.out_dir + f"modes/images/tm={tm}.{fmt}", transparent=True)
        with open(self.out_dir + f"modes/pkl/tm={tm}.pkl", "wb") as f:
            pickle.dump((fig, axes), f)
        plt.clf()
        plt.close()

    def create_animation(self, delete: bool = True) -> None:
        filenames = []
        for tm in range(self.lrprt, self.n, self.lrprt):
            if tm + self.lcurr >= self.n:
                break
            fn = self.out_dir + f"snapshots/images/tm={tm}.png"
            if os.path.exists(fn):
                filenames.append(fn)
        with imageio.get_writer(self.out_dir + "animation.gif", mode="I") as writer:
            for fn in filenames:
                image = imageio.imread(fn)
                writer.append_data(image)
        if delete:
            shutil.rmtree(self.out_dir + "snapshots/images/")

    @property
    def summary(self) -> str:
        def format_line(key, value, fmt=""):
            if ":" not in key:
                key += ":"
            if value is None:
                value = "None"
            return f"{key:<{self.width}} {value:>{self.width}{fmt}}"

        lines = [
            "=" * (2 * self.width + 1),
            "Metadata",
            "=" * (2 * self.width + 1),
            format_line("data", f"({self.n}, {self.d})"),
            format_line("h", self.h),
            format_line("lcurr", self.lcurr),
            format_line("lstep", self.lstep),
            format_line("lrprt", self.lrprt),
            format_line("err_th", self.err_th),
            format_line("trunc_th", self.trunc_th),
            "=" * (2 * self.width + 1),
            "",
            "=" * (2 * self.width + 1),
            "Results",
            "=" * (2 * self.width + 1),
            format_line(
                "RMSE", rmse(self.X[self.train_size :], self.Vf[self.train_size :]), ".3f"
            ),
            format_line("MAE", mae(self.X[self.train_size :], self.Vf[self.train_size :]), ".3f"),
            format_line("avg_time", np.nanmean(self.time), ".2e"),
            "=" * (2 * self.width + 1),
        ]

        return "\n".join(lines)

    @property
    def h(self) -> int:
        return self.metadata["h"]

    @property
    def lcurr(self) -> int:
        return self.metadata["lcurr"]

    @property
    def lrprt(self) -> int:
        return self.metadata["lrprt"]

    @property
    def lstep(self) -> int:
        return self.metadata["lstep"]

    @property
    def err_th(self) -> float:
        return self.metadata["err_th"]

    @property
    def trunc_th(self) -> float:
        return self.metadata["trunc_th"]

    @property
    def Vc(self) -> np.ndarray:
        return np.vstack([self.results[f"Vc{i}"] for i in range(self.d)]).T

    @property
    def Vf(self) -> np.ndarray:
        return np.vstack([self.results[f"Vf{i}"] for i in range(self.d)]).T

    @property
    def B_est(self) -> np.ndarray:
        return self.results["B_est"]

    @property
    def time(self) -> np.ndarray:
        return self.results["time"]

    @property
    def rgm_idx(self) -> np.ndarray:
        return self.results["rgm_idx"]

    @property
    def regime(self) -> np.ndarray:
        return self.results["regime"]

    def __logging_time(self, tm: int, time: float) -> None:
        tf = tm + self.lcurr + self.lstep
        te = tm + self.lcurr + self.lstep + self.lrprt
        self.results["time"][tf:te] = time

    def __logging_W(self, tm: int, W: np.ndarray) -> None:
        self.results["W"][tm : tm + self.lcurr] = W

    def __logging_rgm(self, tm: int, rgm_c: Regime) -> None:
        self.results["regime"][tm : tm + self.lcurr] = [rgm_c] * self.lcurr

    def __logging_rgm_idx(self, tm: int, rgm_idx: float) -> None:
        self.results["rgm_idx"][tm : tm + self.lcurr] = rgm_idx

    def __logging_Vc(self, tm: int, Vc: np.ndarray) -> None:
        for i in range(self.d):
            self.results[f"Vc{i}"][tm + self.lcurr - len(Vc) : tm + self.lcurr] = Vc[:, i]

    def __logging_Vf(self, tm: int, Vf: np.ndarray) -> None:
        tf = tm + self.lcurr + self.lstep
        te = tm + self.lcurr + self.lstep + self.lrprt
        if te < self.n:
            for i in range(self.d):
                self.results[f"Vf{i}"][tf:te] = Vf[:, i]

    def __logging_B_est(self, tm: int, B_est: np.ndarray) -> None:
        for t in range(tm, tm + self.lcurr):
            self.results["B_est"][t] = B_est
