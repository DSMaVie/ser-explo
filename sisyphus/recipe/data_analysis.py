import logging
from pathlib import Path

from erinyes.inference.metrics import BalancedEmotionErrorRate, EmotionErrorRate
from erinyes.preprocess.processor import PreproInstructions
from erinyes.preprocess.stats import DataAnalyzer
from sisyphus import Job, Task, tk

logger = logging.getLogger(__name__)


class DataAnalysisJob(Job):
    def __init__(self, pp_result: tk.Path, label_col:str) -> None:
        super().__init__()
        # instructs = PreproInstructions.from_yaml(Path(pp_inst_path))
        self.analyzer = DataAnalyzer(
            Path(pp_result),
            instructs,
            # metrics={
            #     "eer": EmotionErrorRate(),
            #     "beer": BalancedEmotionErrorRate(
            #         classes=instructs.label_encodec.classes
            #     ),
            # },
        )

        self.stats = self.output_var("stats")
        # self.raw_metrics = self.output_var("raw_metrics")

    def run(self):
        self.analyzer.load_data()
        stats = self.analyzer.compute_stats()
        # result = self.analyzer.compute_prior_metrics(priors) #still a bit faulty
        # logger.info(f"got results {result}")

        # self.raw_metrics.set(result)
        self.stats.set(stats.to_string())

    def tasks(self):
        yield Task("run", mini_task=True)
