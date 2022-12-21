from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import torch

from erinyes.util.types import yamlDict


class Metric(ABC):
    def __init__(self) -> None:
        self.reset()

    def track(self, pred: torch.TensorType, true: torch.TensorType):
        self.pred.extend(pred.numpy())
        self.true.extend(true.numpy())

    @abstractmethod
    def calc(self) -> float:
        ...

    def reset(self):
        self.pred = np.empty()
        self.true = np.empty()

    @classmethod
    def from_yaml(cls, **kwargs):
        return cls(**kwargs)


class EmotionErrorRate(Metric):
    def calc(self) -> float:
        assert (
            self.true.shape == self.pred.shape
        ), f"EmotionErrorRate requires same shapes: {self.pred.shape} (pred) != {self.true.shape} (true)"

        errors = self.true != self.pred
        totals = sum(self.true.shape)
        return errors / totals


class Metrics(Enum):
    eer = EmotionErrorRate
