from __future__ import annotations

import pickle as pkl
from pathlib import Path
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt


def serialize_plot(
    plotting_function: Callable[[matplotlib.axes.Axes], matplotlib.axes.Axes],
    save_path: Path,
    size: tuple[float, float] = [6.4, 4.8],
):
    fig, ax = plt.subplots(
        figsize=size
    )  # need to be created at the sametime, therefore callback loesung
    plotting_function(ax)

    with save_path.with_suffix(".pkl").open("wb") as file:
        pkl.dump(fig, file)
    fig.savefig(save_path.with_suffix(".png"))
