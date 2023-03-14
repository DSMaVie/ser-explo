from __future__ import annotations

import logging
from enum import Enum
from fileinput import filename
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from erinyes.util.enums import Split
from erinyes.util.env import Env
from erinyes.util.globals import BASE_SPLIT_FRACTION

logger = logging.getLogger(__name__)


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
        for keys, value in submap.items():
            if entry == value or entry in keys:
                return value
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

        if fold:
            self.__keep.extend(list(self.__fold.keys()))

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
            if isinstance(val, list):
                self.splits.update({split: val})
            else:
                self.splits.update({split: [val]})

        if Split.TRAIN not in self.splits:
            raise ValueError("At least the train split must be set!")

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        tqdm.pandas(desc="Calculating Splits for Dataset.")

        src_col = data[self.src_col]

        def __wrap(x):
            for split, vals in self.splits.items():
                if x in vals:
                    return split.name.lower()
            return None

        data["split"] = src_col.progress_apply(__wrap)

        # check for missing split. apply default split fraction
        for split in Split:
            if split == Split.TRAIN:
                continue
            if split.name.lower() not in data.split:
                sample = (
                    data.query("split == 'train'")
                    .sample(frac=BASE_SPLIT_FRACTION)
                    .index
                )
                data.split.loc[sample] = split.name.lower()
        return data


class AgreementConsolidator:
    def __init__(
        self,
        target: str,
        target_confounder: str | list[str],
        idx_confounder: str | list[str],
    ) -> None:
        self.target = target
        self.idx_comp = (
            idx_confounder if isinstance(idx_confounder, list) else [idx_confounder]
        )
        self.target_comp = (
            target_confounder
            if isinstance(target_confounder, list)
            else [target_confounder]
        )

    @staticmethod
    def agreement(values: list):
        # determine vote and number of voters for that
        logger.info(f"values to aggree on are {values}")
        uniques = np.unique(values, return_counts=True)
        uniques = sorted(zip(*uniques), key=lambda x: x[1], reverse=True)
        vote, count = uniques[0]

        # determine clearity of winning vote
        if len(uniques) > 2:
            clear_winner = count > uniques[1][1]
        else:
            clear_winner = True

        # return vals
        if clear_winner:
            return vote
        return None

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        idx_cols = [col for col in data.columns if "idx" in col]
        grouped_df = data.groupby(by=idx_cols)
        new_df_records = []
        for name, group in tqdm(
            grouped_df,
            desc="calculating agreement between raters",
            total=grouped_df.ngroups,
        ):
            logger.info(
                f"calculating agreement of group {name} with values {group.to_dict()}"
            )
            vote = self.agreement(group[self.target].values)
            if not vote:
                continue  # agreement could not be reached
            rows_of_vote = group.query(f"{self.target} == {vote!r}")

            tcomp_dict = {
                tcomp: rows_of_vote[tcomp].iloc[0] for tcomp in self.target_comp
            }  # picks first if indistigushable
            key_dict = dict(zip(idx_cols, name))
            keycomp_dict = {kcomp: group.iloc[0][kcomp] for kcomp in self.idx_comp}

            new_df_records.append(
                {self.target: vote, **tcomp_dict, **key_dict, **keycomp_dict}
            )

        return pd.DataFrame.from_records(new_df_records)


class FileSplitter:
    def __init__(self, **files: str) -> None:
        base_dir = Env.load().RAW_DIR
        self.files = {
            Split[split.upper()]: next(base_dir.rglob(f"*{file}"))
            for split, file in files.items()
        }
        logger.info(f"found files for the splits at {self.files}")

    def read_files(self):
        maps = {}
        for split, file in self.files.items():
            logger.info(f"reading file {file}")
            with file.open("r") as f:
                lines = f.readlines()
                lines = [l.strip() for l in lines]
                lines = [l.split("/") if "/" in l else l for l in lines]
                if isinstance(lines[0], str):  # no two keys
                    new_entry = {l: split for l in lines}
                    n_keys = 1
                elif isinstance(lines[0], list):
                    new_entry = {(l[0], int(l[1])): split for l in lines}
                    n_keys = 2
                maps.update(new_entry)
        return maps, n_keys

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        maps, n_keys = self.read_files()

        def _retrieve(key: pd.Series | str):
            key_unpacked = key.iloc[0] if len(key) == 1 else tuple(key.values)

            map_entry = maps.get(key_unpacked)
            return map_entry.name.lower() if map_entry else None

        idx_cols = data.filter(like="idx" if n_keys == 2 else "file_idx")
        tqdm.pandas(desc="associating utterances to splits ...")
        split_col = idx_cols.progress_apply(_retrieve, axis=1)

        logger.info("merging split info back into data.")
        data["split"] = split_col
        data = data[~data.split.isna()]
        return data


class AverageConsolidator:
    def __init__(self, classes: list[str]) -> None:
        if len(classes) == 6:
            self.target = np.array(classes)
            self.mode = "Emotion"
            self.classes = np.array(classes)
        elif len(classes) == 3:
            self.target = self.mode = "Sentiment"
            self.classes = np.array(classes)
        else:
            raise ValueError("Only proper MOS classes allowed!")

    def _compound_NofN(self, row: pd.Series) -> str:
        avg_results = [res > 0 for res in row.values]
        avg_results = self.classes[avg_results]
        return ".".join(avg_results)

    def _compound_single_target(self, value: float) -> str:
        cls_idx = 0 if -1 < value < 1 else 1
        cls_idx = int(np.sign(value) * cls_idx + 1)  # to fix range to cls_idx
        return self.classes[cls_idx]

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        # identify indices
        idx_cols = [col for col in data.columns if "idx" in col]

        # identify averages
        avg_results = data.groupby(by=idx_cols).mean()[self.target]
        tqdm.pandas(desc="compounding based on rater average.", total=len(avg_results))
        if self.mode == "Emotion":
            result = avg_results.progress_apply(self._compound_NofN, axis=1)
        else:
            result = avg_results.progress_apply(self._compound_single_target)

        # identify rest of data merge and return
        data_trunc = data.set_index(idx_cols).drop(
            columns=[
                col for col in data.columns if col in self.target or col in self.classes
            ]
        )
        data_trunc = data_trunc.groupby(by=idx_cols).first()
        data_trunc[self.mode] = result
        return data_trunc.reset_index()


class GatherDurations:
    def __init__(self, pth: Path, filetype: str) -> None:
        self.pth = Env.load().RAW_DIR / pth
        self.type = filetype

    def run(self, data: pd.DataFrame):
        # only intended for entire files!
        data["duration"] = data.file_idx.apply(
            lambda idx: librosa.get_duration(filename=self.pth / f"{idx}.{self.type}")
        )
        return data


class TransformStartStopToDurations:
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        data["duration"] = data.end - data.start
        data = data.drop(columns=["start", "end"])
        return data
