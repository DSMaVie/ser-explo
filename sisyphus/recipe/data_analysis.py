import logging
from pathlib import Path

import torch

from erinyes.inference.metrics import BalancedEmotionErrorRate, EmotionErrorRate
from erinyes.preprocess.stats import DataAnalyzer
from sisyphus import Job, Task, tk

logger = logging.getLogger(__name__)


class DataAnalysisJob(Job):
    def __init__(self, pp_result: tk.Path, label_col: tk.Variable) -> None:
        super().__init__()

        self.path_to_data = pp_result
        self.label_col = label_col

        self.stats = self.output_var("stats")
        self.raw_metrics = self.output_var("raw_metrics")

    def run(self):
        label_encodec = torch.load(Path(self.path_to_data.get()) / "label_encodec.pt")

        lab_col = self.label_col.get()
        if isinstance(lab_col, tuple):
            lab_col = lab_col[0]

        analyzer = DataAnalyzer(
            Path(self.path_to_data),
            lab_col,
            metrics=[
                EmotionErrorRate(),
                BalancedEmotionErrorRate(
                    classes=label_encodec.classes, return_per_emotion=True
                ),
            ],
        )
        analyzer.load_data()
        stats = analyzer.compute_stats()

        # breakpoint()
        priors = stats[stats.index.str.contains("prior")]
        result = analyzer.compute_prior_metrics(priors)
        logger.info(f"got results {result}")

        self.raw_metrics.set(result.to_string())
        self.stats.set(stats.to_string())

    def tasks(self):
        yield Task("run", mini_task=True)
