from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from erinyes.inference.metrics import BalancedEmotionErrorRate, EmotionErrorRate, Metric
from erinyes.util.enums import Split
from sisyphus import Job, tk


class UtteranceLevelDecisionJob(Job):
    def __init__(self, path_to_inferences: tk.Path, class_labels: tk.Variable) -> None:
        super().__init__()

        self.path_to_inferences = Path(path_to_inferences)

        self.decisions = self.output_path("decisions", directory=True)
        self.classes = class_labels.get()

        self.result = self.output_var("results.txt")

    def run(self):
        self.decisions = {}
        for path in self.path_to_inferences.rglob("*.txt"):
            idx_split = path.parts.index("inferences") + 1
            split = path.parts[idx_split]
            idx = path.parts[idx_split + 1 :]

            # remove ending and join
            idx[-1] = idx[-1].split(".")[0]
            idx = "/".join(idx)

            # read and compute results
            with path.open("r") as file:
                line = file.readline()
                true, logits = line.split(";")

                logits = logits.split(",")
                pred = self.decide(logits)

                self.decisions.update(
                    {"idx": idx, "true": true, "pred": pred, "split": split}
                )

        self.decisions = pd.DataFrame.from_records(self.decisions)
        self.metrics = [
            EmotionErrorRate(),
            BalancedEmotionErrorRate(classes=self.classes),
        ]
        self.decisions.to_csv(Path(self.output_path) / "decisions.csv")

    def compute_metrics(self):
        results = {}
        for metric, split in itertools.product(self.metrics, Split):
            dec = self.decisions.query("split = @split")

            metric.track(dec.pred.values, dec.true.values)
            results.update(
                {
                    "split": split,
                    "metric": metric.__class__.__name__,
                    "value": metric.calc(),
                }
            )
            metric.reset()

        results = pd.DataFrame.from_records(results)
        results = results.pivot(index="metric", columns="split", values="values")
        self.result.set(str(results))


class ArgMaxDecision(UtteranceLevelDecisionJob):
    def decide(logits: list[float]) -> int:
        return np.argmax(logits)
