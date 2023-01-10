from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import torch


class DecisionMaker(ABC):
    def __call__(self, batch_of_preds: torch.TensorType | np.ndarray):
        out = []
        for pred in batch_of_preds:
            dec = list(self.generate_decisions(pred))
            out.extend(dec)
        return out

    @abstractmethod
    def generate_decisions(self, pred: torch.TensorType | np.ndarray):
        ...


class OneOfN(DecisionMaker):
    def generate_decisions(self, pred: torch.TensorType | np.ndarray):
        if isinstance(pred, torch.TensorType):
            pred = pred.numpy()
        return np.argmax(pred, axis=-1)


class NOfN(DecisionMaker):
    def __init__(self, threshold: float) -> None:
        assert 0 < threshold < 1, "threshold must be between 0 and 1"
        self.thresh = threshold
        super().__init__()

    def generate_decisions(self, pred: torch.TensorType | np.ndarray):
        if isinstance(pred, torch.TensorType):
            pred = pred.numpy()

        for idx, p in enumerate(pred):
            if p > self.thresh:
                yield idx


class DecisionMakers(Enum):
    OneOfN: OneOfN
    NOfN: NOfN
