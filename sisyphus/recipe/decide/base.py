import itertools
import logging
from pathlib import Path

import pandas as pd

from erinyes.inference.metrics import (BalancedEmotionErrorRate,
                                       EmotionErrorRate)
from erinyes.util.enums import Split
from sisyphus import Job, Task, tk

logger = logging.getLogger(__name__)


class UtteranceLevelDecisionJob(Job):
    def __init__(self, path_to_inferences: tk.Path, class_labels: tk.Variable) -> None:
        super().__init__()

        self.path_to_inferences = Path(path_to_inferences)

        self.decisions = self.output_path("decisions", directory=True)
        self.classes = class_labels.get()

        self.result = self.output_path("results", directory=True)

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
                    {"idx": idx, "true": int(true), "pred": pred, "split": split}
                )

        dec_frame = pd.DataFrame.from_records(dec_list)

        logger.info(f"got decisisons {dec_frame.head().to_string()}")

        metrics = [
            EmotionErrorRate(),
            BalancedEmotionErrorRate(classes=self.classes, return_per_emotion=True),
        ]
        dec_frame.to_csv(Path(self.decisions) / "decisions.csv")

        # compute_metrics
        results = []
        for metric, split in itertools.product(metrics, Split):
            dec = dec_frame.query(f"split == {split.name.lower()!r}")

            metric.track(dec.pred.values, dec.true.values)

            for metric_name, res in metric.calc().items():
                # if metric_name != "beer_total" and metric_name.startswith("beer"):
                #     breakpoint()
                results.append(
                    {
                        "split": split.name.lower(),
                        "metric": metric_name,
                        "value": res,
                    }
                )
            metric.reset()

        results = pd.DataFrame.from_records(results)
        # results = results.pivot(index="metric", columns="split", values="value")
        logger.info(f"got results {results.to_string()}")
        results.to_csv(Path(self.result) / "metrics.csv", index=None)

    def tasks(self):
        yield Task("run")


class SequenceLevelDecisionJob(Job):
    def __init__(
        self, path_to_inferences: tk.Path, path_to_label_encodec: tk.Path
    ) -> None:
        super().__init__()

        self.path_to_inferences = Path(path_to_inferences)
        self.encodec_path = Path(path_to_label_encodec)

        self.decisions = self.output_path("decisions", directory=True)
        self.result = self.output_path("results", directory=True)

    def run(self):
        self.pre_run()

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
                # breakpoint()
                line = file.readline()
                logger.info(f"found labels {line} at {file.name}")
                labels = [int(true) for true in line[1:-1].split(",")]

                logits = []
                for line in file.readlines():
                    line_logits = [float(lit) for lit in line[:-1].split(",")]
                    logits.append(line_logits)

                pred, true = self.decode(logits, labels)
                for line_idx, (phoneme, emotion) in enumerate(true):
                    dec_list.append(
                        {
                            "idx": idx,
                            "type": "true",
                            "phoneme": phoneme,
                            "emotion": emotion,
                            "split": split,
                            "position": line_idx,
                        }
                    )

                for line_idx, (phoneme, emotion) in enumerate(pred):
                    dec_list.append(
                        {
                            "idx": idx,
                            "type": "pred",
                            "phoneme": phoneme,
                            "emotion": emotion,
                            "split": split,
                            "position": line_idx,
                        }
                    )


        dec_frame = pd.DataFrame.from_records(dec_list)
        dec_frame = self.decide(dec_frame)

        logger.info(f"got decisisons {dec_frame.head().to_string()}")
        dec_frame.to_csv(Path(self.decisions) / "decisions.csv")

        # compute_metrics
        results = []
        for split in Split:
            dec = dec_frame.query(f"split == {split.name.lower()!r}")

            result = self.calculate_metrics(dec)
            results.extend(result)

        results = pd.DataFrame(results)
        logger.info(f"got results {results.to_string()}")
        results.to_csv(Path(self.result) / "metrics.csv", index=None)

    def tasks(self):
        yield Task("run")
