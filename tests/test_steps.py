import pandas as pd
import pytest

from erinyes.preprocess.steps import ConditionalSplitter
from erinyes.util.env import Env


@pytest.fixture
def rav_data():
    env = Env.load()  # make higher order fixture in the future
    return pd.read_csv(env.RAW_DIR / "rav" / "manifest.csv")


def test_splitter(rav_data):
    splitter = ConditionalSplitter(src_col="Actor", train="1..18", val=19, test="20,21,22,23,24")
    data = splitter.run(rav_data)
    assert "split" in data.columns
