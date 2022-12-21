from __future__ import annotations

from pathlib import Path

import yaml

from torch.utils.data import DataLoader
from .metrics import Metric, Metrics


class Testbench:
    def __init__(self, monitor: str, metrics: list[Metric]) -> None:
        self.monitor = monitor
        self.metrics = metrics

    @classmethod
    def from_yaml(cls, pth_to_yml_file: Path):
        with pth_to_yml_file.open("r") as file:
            yaml_data = yaml.safe_load(file)

        metrics = []
        for metric_name, metric_args in yaml_data["metrics"].items():
            metrics.append(Metrics[metric_name].value(**metric_args))

        return cls(monitor=yaml_data["monitor"], metrics=metrics)

    def test(pth_to_ckpts:Path,test_data: DataLoader):
        ...
