
from pathlib import Path

import pandas as pd

from erinyes.preprocess.preprocessor import PreProcessor
from erinyes.preprocess.steps import EmotionFilter, LabelNormalizer
from erinyes.util.enums import Dataset
from erinyes.util.env import Env
from sisyphus import Job, Task


class PreprocessingJob(Job):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.out_pth = self.output_path(f"{self.dataset.name}_processed.csv")

    def tasks(self):
        yield Task("run")

    def run(self):
        env = Env.load()

        DATA_DIR = Path(env.RAW_DIR / self.dataset.name.lower())

        manifest = pd.read_csv(DATA_DIR / "manifest.csv.gz")

        # steps
        norm = LabelNormalizer(dataset=self.dataset)
        filt = EmotionFilter(dataset=self.dataset)
        # run
        pp = PreProcessor([norm.normalize_dataframe, filt.filter_dataframe])
        manifest = pp.run(manifest)

        manifest.to_csv(self.out_pth.get_path())



# pp as factory for steps.
#  * steps can be encapsulated as single tasks.
# gather code for entire pp
#  * more steps
#  * more dsets
# link outputs to datadir
# document ideas
#   * sis alias
#
