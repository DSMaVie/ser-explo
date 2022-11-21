from __future__ import annotations

from pathlib import Path

import pandas as pd
from fire import Fire

from erinyes.preprocess.preprocessor import PreProcessor
from erinyes.preprocess.steps import normalize_emotions


def main(data_dir: str):
    DATA_DIR = Path(data_dir)

    manifest = pd.read_csv(DATA_DIR / "manifest.csv")

    pp = PreProcessor([normalize_emotions])
    pp.run(manifest)


if __name__ == "__main__":
    Fire(main)
