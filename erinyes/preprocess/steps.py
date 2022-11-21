from time import sleep

import pandas as pd
from tqdm import tqdm


def normalize_emotions(data: pd.DataFrame) -> pd.DataFrame:
    for idx, row in tqdm(
        data.iterrows(), desc="⤷ Normalizing Emotions", leave=False, total=len(data)
    ):
        sleep(0.005)
