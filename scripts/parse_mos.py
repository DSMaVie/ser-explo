import shutil
import urllib.request
from pathlib import Path

import pandas as pd
from fire import Fire
from tqdm import tqdm


def mos_transcript_generator(pth: Path):
    transcripts = list((pth / "Raw" / "Transcript" / "Segmented").rglob("*.txt"))

    # load dataframe
    cols = ["file_idx", "clip_idx", "start", "end", "transcript"]

    for tr_file in transcripts:
        with open(tr_file, "r") as file:
            lines = file.readlines()
            for line in lines:
                line_info = line.split("___")
                yield dict(zip(cols, line_info))


def produce_label_df(pth: Path):
    label_path = pth / "Raw" / "Labels"
    label_df = pd.concat(
        [pd.read_csv(file) for file in label_path.rglob("*.csv")], axis=0
    )
    label_df = pd.concat(
        (
            label_df.filter(like="Answer"),
            label_df.filter(like="Input"),
        ),
        axis=1,
    )

    label_df = label_df.drop(
        pd.concat(
            (
                label_df.filter(like="secret"),
                label_df.filter(like="load"),
                label_df.filter(like="gender"),
            ),
            axis=1,
        ).columns,
        axis=1,
    )

    label_df = label_df.dropna()

    # some inputs have concrete paths before a '/' cutting around that
    def filter_path_like_videoid(id):
        if isinstance(id, int):
            return id
        elif isinstance(id, str):
            return id.split("/")[-1]
        else:
            raise ValueError(f"found neither str nor int var {id} of type {type(id)}")

    tqdm.pandas(desc="filtering out path_ids that are not usable")
    label_df["Input.VIDEO_ID"] = label_df["Input.VIDEO_ID"].progress_apply(
        filter_path_like_videoid
    )

    return label_df.rename(columns=lambda col: col.split(".")[1])


def pull_mos_split_info(dst: Path):
    split_file = "https://raw.githubusercontent.com/A2Zadeh/CMU-MultimodalSDK/master/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI/cmu_mosei_std_folds.py"
    with urllib.request.urlopen(split_file) as response:
        splits = response.read()
        exec(splits)  ## WARNING !!! HERE BE DRAGONS!!!!!
    # this generates three arrays wich get serialized below.
    # be careful and check the link before running this script.
    # this is the fastest version to pull in the info, albeit not the safest.

    with (dst / "train.txt").open("w") as file:
        file.writelines(standard_train_fold)  # type: ignore

    with (dst / "test.txt").open("w") as file:
        file.writelines(standard_test_fold)  # type: ignore

    with (dst / "val.txt").open("w") as file:
        file.writelines(standard_valid_fold)  # type: ignore


def main(src_dir: str, dst_dir: str):
    SRC_DIR = Path(src_dir)
    DST_DIR = Path(dst_dir)

    tr_gen = mos_transcript_generator(SRC_DIR)
    transcript_df = pd.DataFrame.from_records(tr_gen)

    # type conversions
    transcript_df.clip_idx = transcript_df.clip_idx.astype(int)
    transcript_df.start = transcript_df.start.astype(float)
    transcript_df.end = transcript_df.end.astype(float)

    label_df = produce_label_df(SRC_DIR)
    label_df = label_df.rename(columns={"VIDEO_ID": "file_idx", "CLIP_ID": "clip_idx"})

    manifest = pd.merge(transcript_df, label_df, on=["file_idx", "clip_idx"])

    manifest.to_csv(DST_DIR / "manifest.csv", index=False)
    pull_mos_split_info(DST_DIR)

    for file_idx in tqdm(manifest.file_idx.unique(), desc="copying over files"):
        file_loc = next(SRC_DIR.rglob(f"*{file_idx}.wav"))
        shutil.copy(file_loc, DST_DIR / f"{file_idx}.sph")


if __name__ == "__main__":
    Fire(main)
