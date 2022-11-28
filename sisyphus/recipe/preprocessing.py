from pathlib import Path
from sisyphus import Job, Task
from erinyes.preprocess.preprocessor import PreProcessor
from erinyes.preprocess.steps import normalize_emotions

from erinyes.util.enums import Dataset
import pandas as pd

from erinyes.util.env import Env


class PreprocessingJob(Job):
    def __init__(self, dataset:Dataset) -> None:
        self.dataset = dataset

    def tasks(self):
        yield Task("run")

    def run(self):
        env = Env.load()

        DATA_DIR = Path(env.DATA_DIR / self.dataset.name)

        manifest = pd.read_csv(DATA_DIR / "manifest.csv")

        pp = PreProcessor([normalize_emotions])
        manifest = pp.run(manifest)

        manifest.to_csv(self.output_path(f"{self.dataset.name}_processed.csv"))


# test this
# gather code for entire pp
#  * more steps
#  * more dsets
# link outputs to datadir
# document ideas
# 