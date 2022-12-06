import shutil
from pathlib import Path

import pandas as pd
from fire import Fire
from tqdm import tqdm


def main(src_dir: str, dst_dir: str):
    SRC_DIR = Path(src_dir)
    DST_DIR = Path(dst_dir)
    EMO_DICT = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised",
    }

    data_list = []
    for pth in tqdm(SRC_DIR.rglob("*.wav"), desc="parse and copy data ..."):
        name = pth.stem
        *_, emotion, intensity, statement, repetition, actor = name.split("-")
        data_list.append(
            {
                "file_idx": name,
                "Emotion": EMO_DICT[emotion],
                "Intensity": int(intensity),
                "Actor": int(actor),
                "Gender": "F" if int(actor) // 2 else "M",
                "Statement": "Kids are talking by the door"
                if statement == "01"
                else "Dogs are sitting by the door",
                "Repetition": int(repetition),
            }
        )
        shutil.copy(pth, DST_DIR)

    data = pd.DataFrame.from_records(data_list)
    for col in ["Emotion", "Gender", "Statement"]:
        data[col] = pd.Categorical(data[col])

    data.to_csv(DST_DIR / "manifest.csv", index=False)


if __name__ == "__main__":
    Fire(main)
