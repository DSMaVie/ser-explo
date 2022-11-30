from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from erinyes.util.enums import Dataset, LabelCategory


class LabelNormalizer:
    def __init__(self, dataset=Dataset) -> None:
        self.dataset = dataset
        self.__map = {
            LabelCategory.SENTIMENT: {
                ("pos", "positive"): "Positive",
                ("neut", "neutral"): "Neutral",
                ("neg", "negative"): "Negative",
            },
            LabelCategory.EMOTION: {
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

    def normalize_entry(
        self, str: str, cat: LabelCategory = LabelCategory.EMOTION
    ) -> str | None:
        map = self.__map[cat]
        for keys in map.keys():
            if str in keys:
                return map[keys]
        return None

    def normalize_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        tqdm.pandas(desc="Normalizing Labels...")
        if self.dataset == Dataset.MOS:
            cols = list(LabelCategory)
        elif self.dataset == Dataset.SWBD:
            cols = [LabelCategory.SENTIMENT]
        else:
            cols = [LabelCategory.EMOTION]

        for col in cols:
            data[col.name.lower()] = data[col.name.lower()].progress_apply(
                lambda s: self.normalize_entry(s, cat=col)
            )
        return data


class EmotionFilter:
    def __init__(self, dataset: Dataset, with_neutral: bool = False) -> None:
        if dataset == Dataset.IEM:
            self.__keep = ["Anger", "Excitement", "Happiness", "Sadness", "Frustration"]
            self.__map = {"Excitement": "Happiness"}
        else:
            self.__keep = [
                "Anger",
                "Sadness",
                "Happiness",
                "Fear",
                "Surprise",
                "Disgust",
            ]
            self.__map = {}

        if with_neutral:
            self.__keep.append("Neutral")

    def filter_entry(self, s: str) -> str | None:
        if s not in self.__keep:
            return None

        if s in self.__map.keys():
            return self.__map[s]

        return s

    def filter_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        tqdm.pandas(desc="Filtering and Folding emotions...")
        data.emotion = data.emotion.progress_apply(self.filter_entry)
        return data
