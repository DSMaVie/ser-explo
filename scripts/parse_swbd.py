import re
import shutil
from pathlib import Path

import pandas as pd
from fire import Fire
from tqdm import tqdm


def main(src_dir: str, dst_dir: str):
    SRC_DIR = Path(src_dir)
    DST_DIR = Path(dst_dir)

    # transcripts
    transcript_path = SRC_DIR / "LDC97S62_swb1" / "transcriptions"

    transcript_info = []
    for pth in transcript_path.rglob("*trans.text"):
        with open(pth, "r") as file:
            for line in file.readlines():
                key, start, end, *transcript = line.split(" ")
                audio_id, _, _, clip_id = key.split("-")

                speaker_id = audio_id[-1]
                file_id = "sw0" + audio_id[2:-1]
                transcript = " ".join(transcript).strip("\n")

                transcript_info.append(
                    {
                        "file_idx": file_id,
                        "clip_idx": int(clip_id),
                        "Speaker": speaker_id,
                        "start": float(start),
                        "end": float(end),
                        "transcript": transcript,
                    }
                )

    transcript_df = pd.DataFrame(transcript_info)

    # labels
    label_path = (
        SRC_DIR
        / "LDC2020T14_swb1-sentiment/speech_sentiment_annotations/data/sentiment_labels.tsv"
    )
    label_df = pd.read_csv(
        label_path,
        delimiter="\t",
        header=0,
        names=["file_idx", "start", "end", "label"],
    )

    label_df["clip_idx"] = label_df.file_idx.map(lambda s: s.split("_")[1])
    label_df["file_idx"] = label_df.file_idx.map(lambda s: s.split("_")[0])

    labels = []
    for _, row in tqdm(label_df.iterrows(), "reparsing and expanding rows", total=len(label_df)):
        sub_labels = re.findall(r"(\w+?)-{(.*?)}", row["label"]) # matches: sentiment-{reason}
        for senti, reason in sub_labels:
            labels.append({
                "file_idx": row.file_idx,
                "clip_idx": row.clip_idx,
                "start": row.start,
                "end": row.end,
                "Sentiment": senti,
                "Reason": reason
            })
    label_df = pd.DataFrame.from_records(labels)

    # merge label and transcript info
    label_df["start_int"] = (label_df.start * 100).astype(int)
    label_df["end_int"] = (label_df.end * 100).astype(int)

    transcript_df["start_int"] = (transcript_df.start * 100).astype(int)
    transcript_df["end_int"] = (transcript_df.end * 100).astype(int)

    manifest = pd.merge(
        label_df, transcript_df, on=["file_idx", "start_int", "end_int"], how="left"
    )

    manifest = manifest.drop(columns=[s for s in manifest.columns if s.split("_")[-1] in ["int", "y"]])
    manifest = manifest.rename(columns=lambda s: s[:-2] if s[-1] == "x" and s[-2] == "_" else s)

    # split info
    fold_loc = next(SRC_DIR.rglob("*splits*"))
    folds = {}
    for file in fold_loc.rglob("*.txt"):
        df = pd.read_csv(file, header=1, delimiter="|")
        values = df.drop(index=[0,len(df)-1])[" utterance_id "].str.strip().values.tolist()

        name = file.name.split("_")[0]
        name = "val" if name=="heldout" else name # relabel hold out

        #serialize
        with (DST_DIR / f"{name}.txt").open("w") as file:
            file.writelines("/".join(v.split("_"))+"\n" for v in values)


    # copy to destination
    manifest.to_csv(DST_DIR / "manifest.csv", index=False)
    for file_idx in tqdm(manifest.file_idx.unique(), desc="copying over files"):
        file_loc = next(SRC_DIR.rglob(f"*{file_idx}.sph"))
        shutil.copy(file_loc, DST_DIR / f"{file_idx}.sph")


if __name__ == "__main__":
    Fire(main)
