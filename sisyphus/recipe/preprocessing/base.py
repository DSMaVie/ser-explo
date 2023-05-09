from __future__ import annotations

from pathlib import Path

import pandas as pd

from erinyes.util.env import Env
from sisyphus import Job, Task


class PreprocessingJob(Job):
    def __init__(self, rqmts: dict | None = None) -> None:
        super().__init__()

        self.rqmts = rqmts

        self.label_column = self.output_var("label_column.txt")
        self.utterance_idx = self.output_var("utterance_idx.txt")
        self.out_path = self.output_path("preprocessed_data", directory=True)

    def run(self):
        assert hasattr(self, "processor"), "Job needs to have an initialized processor."

        delayed_args = self.preset()

        src_path = Env.load().RAW_DIR / self.processor.src.name.lower()

        data = pd.read_csv(src_path / "manifest.csv")
        processed_data = self.processor.run_preprocessing(data, delayed_args)
        fe, le = self.processor.extractor_encodec_factory(delayed_args)

        self.processor.serialize(
            data=processed_data,
            feature_extractor=fe,
            label_encodec=le,
            label_targets=self.label_column.get(),
            identifier=self.utterance_idx.get(),
            out_path=Path(self.out_path),
            src_path=src_path,
        )

    def tasks(self):
        yield Task("run", rqmt=self.rqmts)

    def preset(self):
        """presets settings like self.utterance_idx and self.label_column for later use. MUST BE OVERLOADED BY CHILD"""
