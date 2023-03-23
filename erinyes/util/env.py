import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values, find_dotenv
from pynvml import *

logger = logging.getLogger(__name__)


@dataclass
class Env:
    ROOT_DIR: Path
    RAW_DIR: Path
    MODEL_DIR: Path
    INST_DIR: Path

    @classmethod
    @lru_cache(maxsize=None)
    def load(cls):
        str_envs = dotenv_values(find_dotenv())
        return cls(**{k: Path(v) for k, v in str_envs.items()})


def return_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used//1024**2
