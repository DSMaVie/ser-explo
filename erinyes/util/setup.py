from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values, find_dotenv


@dataclass
class Env:
    ROOT_DIR: Path
    DATA_DIR: Path


def setup():
    str_envs = dotenv_values(find_dotenv())
    return Env(**{k: Path(v) for k, v in str_envs.items()})
