import logging
import os

import h5py
import torch
from torch.utils.data import Dataset

from erinyes.util.enums import Split

logger = logging.getLogger(__name__)


class Hdf5Dataset(Dataset):
    def __init__(self, src_path: os.PathLike, split: Split) -> None:
        super().__init__()

        self.src_path = src_path
        self.split = split

    def __len__(self):
        with h5py.File(self.src_path, "r") as file:
            return len(file[self.split.name.lower()].keys())

    def __getitem__(self, idx: str) -> torch.TensorType:
        with h5py.File(self.src_path, "r") as file:

            node = file[f"{self.split.name.lower()}/{idx}"]
            labels = node.attrs["label"]
            features = node[()]
            return features, labels

    def get_indices(self):
        with h5py.File(self.src_path, "r") as file:
            logger.info(
                f"opening file at {self.src_path}. available first lvl keys {list(file.keys())}"
            )
            keys = file[self.split.name.lower()].visit(
                lambda key: key
                if isinstance(file[f"{self.split.name.lower()}/{key}"], h5py.Dataset)
                else None
            )
        return [k for k in keys if k]


# TODO: swbd and mos have file probs -> file splitter is not working properly!
#       -> test with a test!
