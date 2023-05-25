from __future__ import annotations

from .base import SequenceLevelDecisionJob


class ArgMaxSeqDecision(SequenceLevelDecisionJob):
    def decide(self, logits: list[float]) -> list[tuple(str, str)]:
        pass
