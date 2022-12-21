import numpy as np
import pandas as pd
import pytest

from erinyes.preprocess.steps import (AgreementConsolidator,
                                      AverageConsolidator, ConditionalSplitter,
                                      FileSplitter)
from erinyes.util.enums import Split
from erinyes.util.env import Env


@pytest.fixture
def rav_data():
    env = Env.load()  # make higher order fixture in the future
    return pd.read_csv(env.RAW_DIR / "rav" / "manifest.csv")


@pytest.fixture
def swbd_data():
    env = Env.load()  # make higher order fixture in the future
    return pd.read_csv(env.RAW_DIR / "swbd" / "manifest.csv")

@pytest.fixture
def mos_data():
    env = Env.load()  # make higher order fixture in the future
    return pd.read_csv(env.RAW_DIR / "mos" / "manifest.csv")


def test_splitter(rav_data):
    splitter = ConditionalSplitter(
        src_col="Actor", train="1..18", val=19, test="20,21,22,23,24"
    )
    data = splitter.run(rav_data)
    assert "split" in data.columns

def test_file_splitter_swbd(swbd_data):
    file_args = {s.name.lower(): f"swbd/{s.name.lower()}.txt" for s in Split}
    splitter = FileSplitter(**file_args)
    data = splitter.run(swbd_data.iloc[1:100])
    assert "split" in data.columns

    for s in Split:
        assert s.name.lower() in data.split.unique()

def test_file_splitter_mos(mos_data):
    file_args = {s.name.lower(): f"mos/{s.name.lower()}.txt" for s in Split}
    splitter = FileSplitter(**file_args)
    data = splitter.run(mos_data)
    assert "split" in data.columns

    for s in Split:
        assert s.name.lower() in data.split.unique()



def test_agreement(swbd_data):
    agreeer = AgreementConsolidator(target="Sentiment", target_confounder="Reason", idx_confounder=["start", "end", "Speaker", "transcript"])
    agreed_data = agreeer.run(swbd_data)
    assert len(agreed_data) == 50248


def test_agreement_average(mos_data):
    averager = AverageConsolidator(classes=["Happiness", "Anger", "Sadness", "Fear", "Disgust", "Surprise"])
    data = averager.run(mos_data)
    assert len(data) < len(mos_data)