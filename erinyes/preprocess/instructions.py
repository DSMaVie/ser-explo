from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import yaml

from erinyes.features import FeatureExtractor, FeatureExtractors
from erinyes.labels import LabelEncodec
from erinyes.preprocess.steps import PreproFuncs
from erinyes.util.enums import Dataset


@dataclass
class PreproInstructions:
    src: Dataset
    name: str
    steps: list[PreproStep]
    feature_extractor: FeatureExtractor
    label_encodec: LabelEncodec
    label_target: str

    @classmethod
    def from_yaml(cls, pth_to_instructions: Path):
        with pth_to_instructions.open("r") as inst_file:
            str_data = yaml.safe_load(inst_file)

        # compile steps
        steps = []
        for step in str_data["steps"]:
            for k in step.keys():
                if k not in ["name", "args"]:
                    print(step)
                    raise ValueError(
                        f"step description should contain at least the arguments 'args' and 'name'. Only found {step.keys()!s}"
                    )

            step_name = step["name"]
            step_class = PreproFuncs[step_name].value
            step_inst = step_class(**step["args"])

            steps.append(
                PreproStep(
                    name=step_name,
                    func=step_inst.run,
                )
            )

        # compile src, feature_extractor and LabelEncodec
        src = Dataset[str_data["src"]]

        feat_meta = str_data["features"]
        feature_extractor = FeatureExtractors[feat_meta["name"]]
        if feat_meta.get("args"):
            feature_extractor = feature_extractor.value(**feat_meta["args"])
        else:
            feature_extractor = feature_extractor.value()

        label_meta = str_data["labels"]
        label_target = label_meta["target"]
        del label_meta["target"]
        label_encodec = LabelEncodec(**label_meta)

        return cls(
            src=src,
            name=pth_to_instructions.stem,
            steps=steps,
            feature_extractor=feature_extractor,
            label_encodec=label_encodec,
            label_target=label_target,
        )


@dataclass
class PreproStep:
    name: str
    func: Callable[[pd.DataFrame], pd.DataFrame]
