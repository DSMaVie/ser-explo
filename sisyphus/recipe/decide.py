from __future__ import annotations

import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from erinyes.inference.metrics import (BalancedEmotionErrorRate,
                                       EmotionErrorRate)
from erinyes.util.enums import Split
from sisyphus import Job, tk, Task

logger = logging.getLogger(__name__)


class UtteranceLevelDecisionJob(Job):
    def __init__(self, path_to_inferences: tk.Path, class_labels: tk.Variable) -> None:
        super().__init__()

        self.path_to_inferences = Path(path_to_inferences)

        self.decisions = self.output_path("decisions", directory=True)
        self.classes = class_labels.get()

        self.result = self.output_var("results.txt")

    def run(self):
        dec_list = []
        for path in self.path_to_inferences.rglob("*.txt"):
            idx_split = path.parts.index("inferences") + 1
            split = path.parts[idx_split]
            idx = path.parts[idx_split + 1 :]

            # remove ending and join
            idx = "/".join(idx)
            idx = idx.split(".")[0]

            # read and compute results
            with path.open("r") as file:
                line = file.readline()
                logger.info(f"found data {line} at {file.name}")
                true, logits = line.split(";")

                logits = logits.split(",")
                pred = self.decide(logits)

                dec_list.append(
                    {"idx": idx, "true": true, "pred": pred, "split": split}
                )

        dec_frame = pd.DataFrame.from_records(dec_list)
        logger.info(f"got decisisons {dec_frame.head().to_string()}")

        metrics = [
            EmotionErrorRate(),
            BalancedEmotionErrorRate(classes=self.classes),
        ]
        dec_frame.to_csv(Path(self.decisions) / "decisions.csv")

        # compute_metrics
        results = []
        for metric, split in itertools.product(metrics, Split):
            dec = dec_frame.query(f"split == {split.name.lower()!r}")

            metric.track(dec.pred.values, dec.true.values)
            results.append(
                {
                    "split": split.name.lower(),
                    "metric": metric.__class__.__name__,
                    "value": metric.calc(),
                }
            )
            metric.reset()

        results = pd.DataFrame(results)
        results = results.pivot(index="metric", columns="split", values="value")
        logger.info(f"got results {results.to_string()}")
        self.result.set(results.to_string())

    def tasks(self):
        yield Task("run")


class ArgMaxDecision(UtteranceLevelDecisionJob):
    def decide(self, logits: list[float]) -> int:
        return np.argmax(logits)
