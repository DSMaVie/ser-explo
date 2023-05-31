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

    def decide(
        self, dec_frame: pd.DataFrame
    ) -> pd.DataFrame:
        breakpoint()

    def calculate_metrics(
        self, dec_frame: pd.DataFrame
    ) -> list[dict[str, str | float]]:
        pass
