from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd
from tqdm import tqdm

from erinyes.util.enums import Split


class LabelNormalizer:
    def __init__(self, target: str = "Emotion") -> None:
        if target not in ["Sentiment", "Emotion", "Columns"]:
            raise ValueError("target must be in 'Sentiment', 'Emotion' or 'Columns'")

        self.target = target
        self.__map = {
            "Sentiment": {
                ("pos", "positive"): "Positive",
                ("neut", "neutral"): "Neutral",
                ("neg", "negative"): "Negative",
            },
            "Emotion": {
                ("hap", "happy", "happiness"): "Happiness",
                ("exc",): "Excitement",
                ("sad", "sadness"): "Sadness",
                ("ang", "anger", "angry"): "Anger",
                ("dis", "disgust", "disgusted"): "Disgust",
                ("sur", "surprise", "surprised"): "Surprise",
                ("fear", "fearful"): "Fear",
                ("fru",): "Frustration",
                ("oth",): "Other",
                ("calm",): "Calmness",
                ("neu", "neutral", "no emotion"): "Neutral",
            },
        }

    def __normalize_entry(self, entry: str) -> str | None:
        submap = self.__map[self.target]
        for keys in submap:
            if entry in keys:
                return submap[keys]
        return None

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        tqdm.pandas(desc="Normalizing Labels...")
        if self.target != "Columns":
            data[self.target] = data[self.target].progress_apply(self.__normalize_entry)
            return data
        else:
            col_map = {
                sample: self.__map[key]
                for key in self.__map
                for sample in key
                if sample in data.columns
            }
            return data.rename(columns=col_map)


class EmotionFilterNFold:
    def __init__(
        self, keep: list[str] | None = None, fold: dict[str, str] | None = None
    ) -> None:
        self.__keep = keep
        self.__fold = fold

    def __filter_entry(self, s: str) -> str | None:
        if self.__keep:
            if s not in self.__keep:
                return None

        if self.__fold:
            if s in self.__fold.keys():
                return self.__fold[s]

        return s

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        tqdm.pandas(desc="Filtering and Folding emotions...")
        data.Emotion = data.Emotion.progress_apply(self.__filter_entry)
        data = data.dropna()
        return data


class ConditionalSplitter:
    def __init__(self, src_col: str, **kwargs):
        self.splits = {}
        self.src_col = src_col

        for key, val in kwargs.items():
            allowed_kwargs = [s.name.lower() for s in list(Split)]
            if key not in allowed_kwargs:
                raise ValueError(f"key {key} not in {allowed_kwargs}!")

            split = Split[key.upper()]
            vals = self.parse_str_values(val)
            self.splits.update({split: vals})

    @staticmethod
    def parse_str_values(val: str | int):
        vals = None
        no_spread = True
        no_commasep = True

        if isinstance(val, int):
            return [val]

        # check for spread
        deconst_val = val.split("..")
        if len(deconst_val) == 2:
            start, stop = deconst_val
            vals = np.arange(int(start), int(stop) + 1)  # +1 to make the end inclusive
        else:
            no_spread = False

        # check for commasep list
        deconst_val = val.split(",")
        deconst_val = [s.strip() for s in deconst_val]
        if len(deconst_val) > 1:
            vals = [int(s) for s in deconst_val]
        else:
            no_commasep = False

        # check for single value
        if no_commasep and no_spread:
            vals = [int(val)]

        return vals

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        tqdm.pandas(desc="Calculating Splits for Dataset.")

        src_col = data[self.src_col]

        def __wrap(x):
            for split, vals in self.splits.items():
                if x in vals:
                    return split.name.lower()
            return None

        data["split"] = src_col.progress_apply(__wrap)
        return data


class PreproFuncs(Enum):
    normalize_labels = LabelNormalizer
    filter_emotions = EmotionFilterNFold
    produce_conditional_splits = ConditionalSplitter
