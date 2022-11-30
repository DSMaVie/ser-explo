from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values, find_dotenv


@dataclass
class Env:
    ROOT_DIR: Path
    RAW_DIR: Path
    MODEL_DIR: Path


    @classmethod
    def load(cls):
        str_envs = dotenv_values(find_dotenv())
        return cls(**{k: Path(v) for k, v in str_envs.items()})
