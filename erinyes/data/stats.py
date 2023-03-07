from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from erinyes.inference.metrics import Metric
from erinyes.preprocess.instructions import PreproInstructions
from erinyes.util.enums import Split

logger = logging.getLogger(__name__)


class DataAnalyzer:
    def __init__(
        self,
        data_src: Path,
        pp_instructions: PreproInstructions,
        metrics: dict[str, Metric],
    ):
        self.data_src = data_src
        self.pp_instructions = pp_instructions
        self.metrics = metrics

        self.data = pd.read_csv(self.data_src / "manifest.csv", index_col=None)

    def _calc_priors(self, data: pd.DataFrame):
        return data[self.pp_instructions.label_target].value_counts(normalize=True)

    def _calc_words(self, data: pd.DataFrame):
        word_count = data["transcript"].str.split(" ").apply(len)
        uniques = data["transcript"].str.split(" ").apply(np.unique).apply(len)
        vocab_size = len(np.unique(np.stack(data["transcript"].values)))

        return pd.Series(
            {
                "words total": word_count.sum(),
                "avg words per utterance": word_count.mean(),
                "avg unique words per utterance": uniques.mean(),
                "max words per utterance": word_count.max(),
                "max unique words per utterance": uniques.max(),
                "min words per utterance": word_count.min(),
                "min unique words per utterance": uniques.min(),
                "vocabulary size": vocab_size,
                "number utterances": len(data),
            }
        )

    def compute_stats(self):
        priors = {}
        transcript_data = {}
        for split in Split:
            data = self.data.query(f"split == {split.name.lower()!r}")

            priors.update({split.name.lower(): self._calc_priors(data)})
            transcript_data.update({split.name.lower(): self._calc_words(data)})

        return pd.DataFrame(priors), pd.DataFrame(transcript_data)

    def compute_prior_metrics(self, priors: pd.Series):
        split_wise_results = {}

        for split in Split:
            priors = priors.loc[split.name.lower()]
            max_prior_emotions = [
                emo for emo in priors.index if priors.loc[emo] == priors.max()
            ]

            results = {}
            trues = self.data.query(f"split == {split.name.lower()!r}")[
                self.pp_instructions.label_target
            ].to_numpy()

            preds = np.random.choice(
                max_prior_emotions, len(trues)
            )  # produce random choice across max prios

            for mname, metric in self.metrics.items():
                logger.info(f"computing metric: {metric}")
                metric.track(preds, trues)
                results.update({mname: metric.calc()})
                metric.reset()
            split_wise_results.update({split.name.lower(): results})

        return split_wise_results
