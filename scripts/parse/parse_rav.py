import shutil
from pathlib import Path

import pandas as pd
from alive_progress import alive_it
from fire import Fire


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
    for pth in alive_it(SRC_DIR.rglob("*.wav"), title="parse and copy data ..."):
        name = pth.stem
        *_, emotion, intensity, statement, repetition, actor = name.split("-")
        data_list.append(
            {
                "idx": name,
                "emotion": EMO_DICT[emotion],
                "intensity": int(intensity),
                "actor": int(actor),
                "gender": "F" if int(actor) // 2 else "M",
                "statement": "Kids are talking by the door"
                if statement == "01"
                else "Dogs are sitting by the door",
                "repetition": int(repetition),
            }
        )
        shutil.copy(pth, DST_DIR)

    data = pd.DataFrame.from_records(data_list)
    for col in ["emotion", "gender", "statement"]:
        data[col] = pd.Categorical(data[col])
    data = data.set_index("idx")


    data.to_csv(DST_DIR / "manifest.csv")



if __name__ == "__main__":
    Fire(main)
