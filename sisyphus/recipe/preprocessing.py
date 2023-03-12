from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pandas as pd

from erinyes.preprocess.processor import PreproInstructions
from erinyes.preprocess.serialization import serialize_preprocessed_data
from erinyes.util.enums import Dataset
from erinyes.util.env import Env
from sisyphus import Job, Task, tk

logger = logging.getLogger(__name__)


class PreprocessingJob(Job):
    def __init__(
        self,
        pth_to_instructions: tk.Path,
        pth_to_download: tk.Path | None = None,
        cache_data: bool = False,
    ) -> None:
        logger.info(f"loading instructions from {pth_to_instructions}")
        self.instructions = PreproInstructions.from_yaml(
            pth_to_instructions=Path(pth_to_instructions.get_path()),
            pth_to_predownload=Path(pth_to_download.get_path())
            if pth_to_download
            else None,
        )
        self.cache_data = cache_data

        self.out_pth = self.output_path("", directory=True)

    def process_manifest(self):
        manifest_pth = (
            Env.load().RAW_DIR / self.instructions.src.name.lower() / "manifest.csv"
        )
        logger.info(f"loading manifest from {manifest_pth}")
        manifest = pd.read_csv(manifest_pth)

        for step in self.instructions.steps:
            logger.info(f"executing step: {step.name} ...")
            manifest = step.func(manifest)

        logger.info(f"serializing manifest ...")
        out_pth = Path(self.out_pth.get_path())
        manifest.to_csv(out_pth / "manifest.csv", index=False)

    def finalize(self):
        out_pth = Path(self.out_pth.get_path())

        logger.info(f"serializing feature_extractor ...")
        fe = self.instructions.feature_extractor
        with (out_pth / "feature_extractor.pkl").open("wb") as file:
            pickle.dump(fe, file)

        logger.info(f"serializing label_encodec ...")
        le = self.instructions.label_encodec
        with (out_pth / "label_encoder.pkl").open("wb") as file:
            pickle.dump(le, file)

        logger.info(f"serializing data for quick access ...")
        serialize_preprocessed_data(
            out_pth,
            src_path=Env.load().RAW_DIR / self.instructions.src.name.lower(),
            feature_extractor=fe,
            label_encodec=le,
            target_col=self.instructions.label_target,
            use_start_end=self.instructions.src != Dataset.IEM,
        )

    # def transfer_to_local:
    #     self.sh()

    def tasks(self):
        yield Task("process_manifest", mini_task=True)
        yield Task("finalize", mini_task=True)
        if self.cache_data:
            yield Task("transfer_to_local", mini_task=True)
