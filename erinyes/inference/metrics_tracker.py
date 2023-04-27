from __future__ import annotations

import numpy as np
from transformers import EvalPrediction

from erinyes.inference.metrics import Metric


class InTrainingsMetricsTracker:
    def __init__(self, metrics: list[Metric]) -> None:
        self.metrics = metrics

    def __call__(self, p: EvalPrediction) -> dict:
        """adhering to compute_metrics form hf transformers"""
        pred = np.argmax(p.predictions, axis=-1)
        true = p.label_ids

        res_dict = {}
        for metric in self.metrics:
            metric.track(pred, true)
            res_dict.update(metric.calc())
            metric.reset()

        return res_dict
