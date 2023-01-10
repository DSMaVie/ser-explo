from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import yaml

from .metrics import Metric, Metrics

logger = logging.getLogger(__name__)


class Testbench:
    def __init__(self, metrics: dict[str, Metric]) -> None:
        self.metrics = metrics

    @classmethod
    def from_yaml(cls, pth_to_yml_file: Path):
        with pth_to_yml_file.open("r") as file:
            yaml_data = yaml.safe_load(file)

        metrics = []
        for metric_name, metric_args in yaml_data["metrics"].items():
            metrics.update({metric_name: Metrics[metric_name].value(**metric_args)})

        return cls(metrics=metrics)

    def test(self, pth_to_predictions: Path):
        for pred_file in pth_to_predictions.rglob("*.txt"):
            with pred_file.open("r") as file:
                lines = file.readlines()

            lines = [l.split(" --- ") for l in lines]
            lines = [(l[0], l[1].split(",")) for l in lines]

            true, pred = zip(*lines)
            pred = np.argmax(
                pred, axis=-1
            )  # needs to be more sophisticated // breaks for mos_ekman

            res = {}
            for m_name, metric in self.metrics.items():
                metric.track(pred, true)
                res = metric.calc()
                res.update({m_name: res})

            return res
