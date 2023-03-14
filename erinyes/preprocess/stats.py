from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# from erinyes.inference.metrics import Metric
from erinyes.util.enums import Split

logger = logging.getLogger(__name__)


class DataAnalyzer:
    def __init__(
        self,
        data_src: Path,
        label_col:str,
        # metrics: dict[str, Metric],
    ):
        self.data_src = data_src
        self.label_col = label_col
        # self.metrics = metrics

    def load_data(self):
        self.data = pd.read_csv(self.data_src / "manifest.csv", index_col=None)

    def _prior_stats(self, data: pd.DataFrame):
        assert hasattr(self, "data"), "data needs to be loaded first!"
        # TODO: handle mhe case
        priors = data[self.label_col].value_counts(normalize=True)
        return priors.add_prefix("prior_")

    def _time_stats(self, data: pd.DataFrame):
        if "start" in data.columns:
            dur = data["end"] - data["start"]
        else:
            dur = data["duration"]

        return pd.Series(
            {
                "total duration": dur.sum() / 60 / 60,
                "avg duration per utterance": dur.mean(),
                "max duration per utterance": dur.max(),
                "min duration per utterance": dur.min()
            }
        )

    def _word_stats(self, data: pd.DataFrame):
        assert hasattr(self, "data"), "data needs to be loaded first!"

        text_keyword = "transcript" if "transcript" in data.columns else "Statement"

        word_lists= data[text_keyword].str.split(" ").dropna()
        word_count = word_lists.apply(len)

        return pd.Series(
            {
                "words total": word_count.sum(),
                "avg words per utterance": word_count.mean(),
                "max words per utterance": word_count.max(),
                "min words per utterance": word_count.min(),
                "number utterances": len(data),
            }
        )

    def _vocab_stats(self, data: pd.DataFrame):
        assert hasattr(self, "data"), "data needs to be loaded first!"

        text_keyword = "transcript" if "transcript" in data.columns else "Statement"
        uniques = data[text_keyword].str.split(" ").dropna().values
        uniques = [set(t) for t in uniques]
        unique_lens = [len(s) for s in uniques]

        return pd.Series(
            {
                "avg unique words per utterance": np.mean(unique_lens),
                "max unique words per utterance": np.max(unique_lens),
                "min unique words per utterance": np.min(unique_lens),
                "vocabulary size": len(set.union(*uniques)),
            }
        )

    def compute_stats(self):
        stats = {
            "total": pd.concat(
                [
                    self._time_stats(self.data),
                    self._prior_stats(self.data),
                    self._word_stats(self.data),
                    self._vocab_stats(self.data),
                ]
            )
        }

        for split in Split:
            data = self.data.query(f"split == {split.name.lower()!r}")

            if not len(data):
                continue

            stats.update(
                {
                    split.name.lower(): pd.concat(
                        [
                            self._time_stats(data),
                            self._prior_stats(data),
                            self._word_stats(data),
                            self._vocab_stats(data),
                        ]
                    )
                }
            )

        return pd.DataFrame(stats)

    # def compute_prior_metrics(self, priors: pd.Series):
    #     split_wise_results = {}

    #     for split in Split:
    #         priors = priors[split.name.lower()]
    #         max_prior_emotions = [
    #             emo for emo in priors.index if priors.loc[emo] == priors.max()
    #         ]
    #         priors.index.str.removeprefix("prior_")

    #         results = {}
    #         trues = self.data.query(f"split == {split.name.lower()!r}")[
    #             self.pp_instructions.label_target
    #         ].to_numpy()

    #         preds = np.random.choice(
    #             max_prior_emotions, len(trues)
    #         )  # produce random choice across max prios

    #         for mname, metric in self.metrics.items():
    #             logger.info(f"computing metric: {metric}")
    #             metric.track(preds, trues)
    #             results.update({mname: metric.calc()})
    #             metric.reset()
    #         split_wise_results.update({split.name.lower(): results})

    #     return split_wise_results
