from __future__ import annotations

from typing import Callable

import pandas as pd
from tqdm import tqdm

preProcessorStep = Callable[[pd.DataFrame], pd.DataFrame]


class PreProcessor:
    def __init__(self, steps: list[preProcessorStep]) -> None:
        self.steps = steps

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        for step in tqdm(self.steps, desc="Preprocessing"):
            data = step(data)
        tqdm.write("test", end="sth")
        return data
