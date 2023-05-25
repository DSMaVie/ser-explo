from __future__ import annotations

import numpy as np

from .base import UtteranceLevelDecisionJob


class ArgMaxDecision(UtteranceLevelDecisionJob):
    def decide(self, logits: list[float]) -> int:
        return np.argmax(logits)
