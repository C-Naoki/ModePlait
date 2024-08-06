import os

import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.figure import Figure

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 15
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["xtick.major.width"] = 1.0
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["ytick.major.width"] = 1.0
plt.rcParams["axes.linewidth"] = 1.0


def change2darkmode(type: str = "notion") -> None:
    color_code = "#191919" if type == "notion" else "#323232"
    plt.rcParams["text.color"] = "white"
    plt.rcParams["axes.titlecolor"] = "white"
    plt.rcParams["xtick.color"] = "white"
    plt.rcParams["axes.labelcolor"] = "white"
    plt.rcParams["ytick.color"] = "white"
    plt.rcParams["axes.facecolor"] = color_code
    plt.rcParams["axes.edgecolor"] = "white"
    plt.rcParams["figure.facecolor"] = color_code
    plt.rcParams["figure.edgecolor"] = "white"
    plt.rcParams["legend.facecolor"] = "dimgray"
    plt.rcParams["legend.labelcolor"] = "white"
    plt.rcParams["axes.prop_cycle"] = cycler(
        "color",
        ["#8dd3c7", "#feffb3", "#bfbbd9", "#fa8174", "#81b1d2", "#fdb462", "#b3de69", "#bc82bd", "#ccebc4", "#ffed6f"],
    )
