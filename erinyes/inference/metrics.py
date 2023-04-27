from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Metric(ABC):
    def __init__(self) -> None:
        self.reset()

    def track(self, pred: np.ndarray, true: np.ndarray):
        assert (
            len(pred.shape) == 1 and len(true.shape) == 1
        ), f"pred and true should only have one dimension but have shape {pred.shape} (pred) and {true.shape} (true)"
        if self.pred is not None:
            self.pred = np.stack([self.pred, pred], axis=0)
        else:
            self.pred = pred
        if self.true is not None:
            self.true = np.stack([self.true, true], axis=0)
        else:
            self.true = true

    @abstractmethod
    def calc(self) -> dict[str, float]:
        ...

    def reset(self):
        self.pred = None
        self.true = None

    @classmethod
    def from_yaml(cls, **kwargs):
        return cls(**kwargs)


class EmotionErrorRate(Metric):
    def calc(self) -> float:
        assert (
            self.true.shape == self.pred.shape
        ), f"EmotionErrorRate requires same shapes: {self.pred.shape} (pred) != {self.true.shape} (true)"

        errors = np.sum(np.not_equal(self.true, self.pred))
        totals = np.sum(self.true.shape)
        return {"eer": errors / totals}


class BalancedEmotionErrorRate(Metric):
    def __init__(self, classes: list[str], return_per_emotion: bool = False) -> None:
        super().__init__()

        self.classes = classes
        self.return_per_emotion = return_per_emotion

    def calc(self) -> dict[str, float]:
        assert (
            self.true.shape == self.pred.shape
        ), f"EmotionErrorRate requires same shapes: {self.pred.shape} (pred) != {self.true.shape} (true)"

        beers = {}
        for idx, val in enumerate(self.classes):
            emo_mask = self.true == idx
            trues = self.true[emo_mask]
            preds = self.pred[emo_mask]

            errors = np.sum(np.not_equal(trues, preds))
            totals = np.sum(emo_mask.shape)
            beers.update({f"beer_{val}": errors / totals})

        total = {"beer_total": np.mean(list(beers.values()))}

        if self.return_per_emotion:
            beers.update(total)
            return beers
        return total
