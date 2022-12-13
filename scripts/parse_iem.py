import shutil
from pathlib import Path

import pandas as pd
from fire import Fire
from tqdm import tqdm


def generate_iem_metadata(pth: Path):
    for file in tqdm(list(pth.rglob("*.txt")), desc="Unpacking IEM Metadata"):
        if file.parent.name != "EmoEvaluation":
            continue
        file_info = file.stem.split("_")

        sesh, gender = file_info[0][-2:]
        mode, idx = file_info[1].split("0")
        rep = file_info[2] if mode == "script" else None

        with file.open("r") as reader:
            lines = reader.readlines()
        for line in lines:
            line = line.strip().split("\t")
            if len(line) != 4:
                continue
            full_idx = line[1]
            emotion = line[2]
            start, end = line[0].strip("[]").split(" - ")

            yield {
                "file_idx": full_idx,
                "Session": int(sesh),
                "Recording": idx,
                "Gender": gender,
                "Repetition": rep,
                "Emotion": emotion,
                "start": float(start),
                "end": float(end),
            }


def generate_iem_transcript_data(pth: Path):
    for tr_file in tqdm(list(pth.rglob("*.txt")), desc="Analyzing Transcription Files"):
        if tr_file.parts[-2] != "transcriptions":
            continue

        lines = []
        with tr_file.open("r") as f:
            lines = f.readlines()

        for line in tqdm(lines, desc=f"Reading contents of {tr_file}", leave=False):
            line = line.split()
            if line[0][0] in ["M", "F"]:
                continue  # skip intermissions

            start, end = line[1].strip("[]:").split("-")
            if len(line[2:]) != 1:
                text = " ".join(line[2:]).strip()
            else:
                text = line[2]
            yield {"file_idx": line[0], "start": start, "end": end, "transcript": text}


def main(src_dir: str, dst_dir: str):
    SRC_DIR = Path(src_dir)
    DST_DIR = Path(dst_dir)

    # labels
    iem_generator = generate_iem_metadata(SRC_DIR)
    lab_df = pd.DataFrame.from_records(iem_generator)
    lab_df.Recording = pd.Categorical(lab_df.Recording)
    lab_df.Gender = pd.Categorical(lab_df.Gender)
    lab_df.Repetition = pd.Categorical(lab_df.Repetition)
    lab_df.Emotion = pd.Categorical(lab_df.Emotion)

    # transcripts
    transcript_generator = generate_iem_transcript_data(SRC_DIR)
    tr_df = pd.DataFrame.from_records(transcript_generator)
    tr_df.start = tr_df.start.astype(float)
    tr_df.end = tr_df.end.astype(float)

    # merged manifest
    manifest = pd.merge(tr_df, lab_df, on=["file_idx", "start", "end"])
    manifest.to_csv(DST_DIR / "manifest.csv", index=False)

    for f_idx in tqdm(manifest.file_idx, desc="Copying audio files"):
        pth = next(SRC_DIR.rglob(f"*{f_idx}.wav"))
        shutil.copy(pth, DST_DIR / pth.name)


if __name__ == "__main__":
    Fire(main)
