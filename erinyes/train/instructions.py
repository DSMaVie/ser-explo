

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingsInstructions:


    @classmethod
    def from_yaml(cls, pth_to_arch_params: Path, pth_to_train_instructs: Path):
        ...