from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from erinyes.inference.metrics import BalancedEmotionErrorRate, EmotionErrorRate

from .base import SequenceLevelDecisionJob


class ArgMaxSeqDecision(SequenceLevelDecisionJob):
    def pre_run(self):
        self.label_encodec = torch.load(self.encodec_path / "label_encodec.pt")
        self.metrics = [
            EmotionErrorRate(),
            BalancedEmotionErrorRate(
                classes=self.label_encodec.classes, return_per_emotion=True
            ),
        ]

    def decode(self, logits: np.ndarray, labels: np.ndarray) -> list[tuple(str, str)]:
        trues = self.label_encodec.decode(labels)

        probs = np.exp(logits) / (np.exp(logits) + 1)
        pred = self.label_encodec.decode(np.argmax(probs, axis=1))

        return pred, trues

    def decide(self, dec_frame: pd.DataFrame) -> pd.DataFrame:
        # select bny mode (most common occurence)
        decisions = (
            dec_frame.filter(["idx", "emotion", "split", "type"])
            .query("emotion != 'No Emotion'")
            .groupby(["idx", "type"])
            .agg({"emotion": pd.Series.mode, "split": "first"})
            .reset_index()
        )
        # shuffle and drop duplicates for random select of duplicates
        decisions = (
            decisions.explode("emotion")
            .sample(frac=1)
            .drop_duplicates(subset=["idx", "type"])
            .reset_index(drop=True)
        )

        return decisions

    def calculate_metrics(
        self, dec_frame: pd.DataFrame
    ) -> list[dict[str, str | float]]:
        split = dec_frame.split.iloc[0]

        dec_frame_pivot = dec_frame.pivot(index="idx", columns="type", values="emotion")
        trues = dec_frame_pivot.true.values
        preds = dec_frame_pivot.pred.values

        results = []
        for metric in self.metrics:
            metric.track(preds, trues)

            for metric_name, res in metric.calc().items():
                results.append(
                    {
                        "split": split,
                        "metric": metric_name,
                        "value": res,
                    }
                )
            metric.reset()

        return results
