from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pandas as pd
import seaborn as sns

from erinyes.analysis import serialize_plot
from sisyphus import Job, Task, tk
from sisyphus.task import Task


class TrainAnalysisJob(Job):
    def __init__(self, train_result: tk.Path, model_class: tk.Variable) -> None:
        super().__init__()

        self.train_result_path = Path(train_result)
        self.model_class = model_class

        self.out = self.output_path("training_results", directory=True)

    def load_info(self):
        meta_data = {}
        with (self.train_result_path.parent.parent / "info").open("r") as file:
            for line in file.readlines():
                if line.startswith("INPUT:"):
                    _, path = line.split(" ")
                    path = Path(path)

                    if "preprocessing" in path.parts:
                        # line is data path
                        pp_index = path.parts.index("preprocessing")
                        meta_data["data_condition"] = path.parts[pp_index + 1]
                    elif "download_pt_model" in path.parts:
                        # line is base_model
                        meta_data["base_model"] = path.name

        meta_data["model"] = self.model_class.get().__name__

        with (self.train_result_path / "trainer_state.json").open("r") as file:
            raw_state = json.load(file)

        return pd.DataFrame.from_records(data=raw_state["log_history"]), meta_data

    def produce_loss_plot(self, data: pd.DataFrame, meta_data:dict):
        validation_data = data.filter(["epoch", "eval_loss"]).dropna()
        validation_data["kind"] = "Validation"
        validation_data = validation_data.rename(columns={"eval_loss": "loss"})

        train_data = data.filter(["epoch", "loss"]).dropna()
        train_data["kind"] = "Training"

        data = pd.concat(
            [train_data, validation_data], axis=0, ignore_index=True, join="outer"
        )

        def plotter(ax):
            ax = sns.lineplot(
                data=data,
                x="epoch",
                y="loss",
                hue="kind",
                style="kind",
                ax=ax,
                estimator=None,
            )
            ax.set_title(f"{meta_data['data_condition']} | {meta_data['base_model']}")

        serialize_plot(plotter, Path(self.out) / "loss")

    def produce_lr_plot(self, data: pd.DataFrame, meta_data:dict):
        data = data.filter(["epoch", "learning_rate"]).dropna()

        def plotter(ax):
            ax = sns.lineplot(
                data=data,
                x="epoch",
                y="learning_rate",
                ax=ax,
                estimator=None,
            )
            ax.set_title(f"{meta_data['data_condition']} | {meta_data['base_model']}")

        serialize_plot(plotter, Path(self.out) / "learning_rate")

    def run(self):
        data, meta_data = self.load_info()
        self.produce_loss_plot(data, meta_data)
        self.produce_lr_plot(data, meta_data)

    def tasks(self) -> Iterator[Task]:
        yield Task("run")
