from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Protocol, TypeVar

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from erinyes.data.labels import LabelEncodec
from erinyes.util.enums import Dataset

logger = logging.getLogger(__name__)


class FeatureExtractor(Protocol):
    def extract(
        self, pth_to_data: Path, start: float = 0, duration: float = None
    ) -> np.ndarray:
        ...


class PreproFunc(Protocol):
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        ...


RecipeType = TypeVar("RecipeType", PreproFunc, LabelEncodec, FeatureExtractor)


@dataclass
class PreproTask(Generic[RecipeType]):
    name: str
    instance: RecipeType
    args: dict | None = None

    def create_instance(self):
        if self.args is not None:
            return self.instance(**self.args)
        return self.instance()


@dataclass
class Preprocessor:
    src: Dataset
    name: str
    steps: list[PreproTask[PreproFunc]]
    feature_extractor: PreproTask[FeatureExtractor]
    label_encodec: PreproTask[LabelEncodec]

    def run_preprocessing(self, data: pd.DataFrame):
        for step in self.steps:
            logger.info(f"instantiating step {step.name} with args {step.args}")
            processor = step.create_instance()

            logger.info(f"running data through processor {step.name}")
            data = processor.run(data)

        logger.info(f"processing of manifest complete.")
        return data

    def extractor_encodec_factory(self):
        logger.info("creating of feature extractor and label encodec.")
        extractor = self.feature_extractor.create_instance()
        encodec = self.label_encodec.create_instance()
        return extractor, encodec

    def serialize(
        self,
        data: pd.DataFrame,
        feature_extractor: FeatureExtractor,
        label_encodec: LabelEncodec,
        label_target: str,
        identifier: str | list[str],
        out_path: Path,
        src_path: Path,
    ):
        if isinstance(identifier, str):
            identifier = [identifier]

        for col in ["split", label_target, *identifier]:
            if col not in data.columns:
                raise ValueError(f"could not find {col} in columns of manifest.")

        with h5py.File(out_path / "processed_data.h5", "w") as file:
            for _, row in tqdm(data.iterrows(), "Extracting and Encoding for Model..."):
                start = row.start if "start" in row.index else 0
                duration = (
                    row.end - row.start  # case start and end marker
                    if "start" in row.index and "end" in row.index
                    else None
                )
                pth_to_file = next(src_path.rglob(f"*{row.file_idx}.*"))

                logger.info(f"extracting from {pth_to_file}")
                features = feature_extractor.extract(
                    pth_to_data=pth_to_file,
                    start=start,
                    duration=duration,
                )
                label = label_encodec.encode(row[label_target])

                groupkeys = "/".join([row.split] + [str(row[k]) for k in identifier])
                grp = file.create_group(groupkeys)
                grp.create_dataset("features", data=features)
                grp.create_dataset(
                    "label", data=(label,) if not isinstance(label, np.ndarray) else label
                )

        data.to_csv(out_path / "manifest.csv")
        torch.save(feature_extractor, out_path / "feature_extractor.pt")
        torch.save(label_encodec, out_path / "label_encodec.pt")
