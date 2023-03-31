from __future__ import annotations

from transformers import EvalPrediction

from erinyes.inference.metrics import Metric


class InTrainingsMetricsTracker:
    def __init__(self, metrics: list[Metric]) -> None:
        self.metrics = {m.__class__.__name__: m for m in metrics}

    def __call__(self, p: EvalPrediction) -> dict:
        """adhering to compute_metrics form hf transformers"""
        pred = p.predictions
        true = p.label_ids

        res_dict = {}
        for name, metric in self.metrics.items():
            metric.track(pred, true)
            res_dict.update({name, metric.calc()})
            metric.reset()

        return res_dict
