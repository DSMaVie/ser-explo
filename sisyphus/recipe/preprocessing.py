from pathlib import Path

import pandas as pd

from erinyes.preprocess.preprocessor import PreProcessor
from erinyes.preprocess.steps import normalize_emotions
from erinyes.util.enums import Dataset
from erinyes.util.env import Env
from sisyphus import Job, Task


class PreprocessingJob(Job):
    def __init__(self, dataset:Dataset) -> None:
        self.dataset = dataset
        self.out_pth = self.output_path(f"{self.dataset.name}_processed.csv")

    def tasks(self):
        yield Task("run")

    def run(self):
        env = Env.load()

        DATA_DIR = Path(env.DATA_DIR / "raw" / self.dataset.name.lower())

        manifest = pd.read_csv(DATA_DIR / "manifest.csv.gz")

        pp = PreProcessor([normalize_emotions])
        manifest = pp.run(manifest)

        manifest.to_csv(self.out_pth.get_path())


# test this
# gather code for entire pp
#  * more stepswhich s
#  * more dsets
# link outputs to datadir
# document ideas
#   * sis alias
#