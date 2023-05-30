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

    def decide(self, true: list[int], preds: np.ndarray) -> list[tuple(str, str)]:
        pass

    def calculate_metrics(
        self, dec_frame: pd.DataFrame
    ) -> list[dict[str, str | float]]:
        pass
