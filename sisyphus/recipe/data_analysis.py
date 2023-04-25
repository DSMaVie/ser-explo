import logging
from pathlib import Path

# from erinyes.inference.metrics import BalancedEmotionErrorRate, EmotionErrorRate
from erinyes.preprocess.stats import DataAnalyzer
from sisyphus import Job, Task, tk

logger = logging.getLogger(__name__)


class DataAnalysisJob(Job):
    def __init__(self, pp_result: tk.Path, label_col: tk.Variable) -> None:
        super().__init__()

        self.path_to_data = pp_result
        self.label_col = label_col

        self.stats = self.output_var("stats")
        # self.raw_metrics = self.output_var("raw_metrics")

    def run(self):
        analyzer = DataAnalyzer(
            Path(self.path_to_data),
            self.label_col.get(),
            # metrics={
            #     "eer": EmotionErrorRate(),
            #     "beer": BalancedEmotionErrorRate(
            #         classes=instructs.label_encodec.classes
            #     ),
            # },
        )
        analyzer.load_data()
        stats = analyzer.compute_stats()
        # result = analyzer.compute_prior_metrics(priors) #still a bit faulty
        # logger.info(f"got results {result}")

        # self.raw_metrics.set(result)
        self.stats.set(stats.to_string())



    def tasks(self):
        yield Task("run", mini_task=True)
