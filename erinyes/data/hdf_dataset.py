import logging
import os

import h5py
import torch
from torch.utils.data import Dataset

from erinyes.util.enums import Split

logger = logging.getLogger(__name__)


class Hdf5Dataset(Dataset):
    def __init__(
        self, src_path: os.PathLike, split: Split, load_data: bool = False
    ) -> None:
        super().__init__()
        logger.info(f"loading data from {src_path} for split {split}")

        self.src_path = src_path
        self.split = split

    def __len__(self):
        with h5py.File(self.src_path, "r") as file:
            return len(file[self.split.name.lower()].keys())

    def __getitem__(self, idx: str) -> torch.TensorType:
        with h5py.File(self.src_path, "r") as file:
            node = file[f"{self.split.name.lower()}/{idx}"]

            logger.debug(f"retrieving node {self.split.name.lower()}/{idx}")
            labels = torch.Tensor(node["label"][()])
            features = torch.Tensor(node["features"][()])

            return features, labels

    @property
    def available_indices(self):
        keys = []
        with h5py.File(self.src_path, "r") as file:
            logger.info(
                f"opening file at {self.src_path}. available first lvl keys {list(file.keys())}"
            )
            curr_block = file[self.split.name.lower()]
            curr_block.visit(
                lambda key: keys.append(key)
                if isinstance(curr_block[key], h5py.Group)
                else None
            )
        return keys
