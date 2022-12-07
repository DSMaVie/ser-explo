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

# splits for swbd
# fix ravdess
# iem
# mos
#   emotion
#   sentiment


class PreproFuncs(Enum):
    normalize_labels = LabelNormalizer
    filter_emotions = EmotionFilterNFold
    produce_conditional_splits = ConditionalSplitter
    consolidate_per_agreement = AgreementConsolidator
