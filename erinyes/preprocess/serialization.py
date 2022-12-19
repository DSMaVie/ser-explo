from __future__ import annotations

from pathlib import Path

import h5py
import pandas as pd
from tqdm import tqdm

from erinyes.data.features import FeatureExtractor
from erinyes.data.labels import LabelEncodec


def serialize_preprocessed_data(
    out_pth: str | Path,
    src_path: str | Path,
    feature_extractor: FeatureExtractor,
    label_encodec: LabelEncodec,
    target_col: str,
):
    if isinstance(out_pth, str):
        out_pth = Path(out_pth)

    if isinstance(src_path, str):
        src_path = Path(src_path)

    manifest = pd.read_csv(out_pth / "manifest.csv")

    with h5py.File(out_pth / "processed_data.h5", "w") as file:
        for _, row in tqdm(manifest.iterrows(), "Extracting and Encoding for Model..."):
            start = row.start if "start" in row.index else 0
            duration = (
                row.end - row.start
                if "start" in row.index and "end" in row.index
                else None
            )
            pth_to_file = next(src_path.rglob(f"*{row.file_idx}.*"))

            features = feature_extractor.extract(
                pth_to_data=pth_to_file,
                start=start,
                duration=duration,
            )
            labels = label_encodec.encode(row[target_col])

            keys = "/".join([str(row[k]) for k in row.index if "idx" in k])
            dset = file.create_dataset(f"{row.split}/{keys}", data=features)
            dset.attrs["label"] = labels
