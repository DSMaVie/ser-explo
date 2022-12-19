from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import yaml
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler

from erinyes.data.features import FeatureExtractors
from erinyes.data.hdf_dataset import Hdf5Dataset
from erinyes.data.labels import LabelEncodec
from erinyes.data.util import collate_with_pack_pad_to_batch
from erinyes.models import Models
from erinyes.util.enums import Split
from erinyes.util.types import yamlDict
from erinyes.train.trainer import Trainer

@dataclass
class TrainingsInstructions:
    model: nn.Module
    data: DataLoader
    trainer: Trainer

    @classmethod
    def from_yaml(
        cls,
        pth_to_arch_params: Path,
        pth_to_train_instructs: Path,
        pth_to_pp_output: Path,
    ):
        with pth_to_arch_params.open("r") as arch_file:
            arch_data = yaml.safe_load(arch_file)

        with pth_to_train_instructs.open("r") as train_file:
            train_data = yaml.safe_load(train_file)

        model = cls._get_model(
            model_name=train_data["model"],
            feature_extractor=pickle.load(pth_to_pp_output / "feature_extractor.pkl"),
            label_encodec=pickle.load(pth_to_pp_output / "label_encodec.pkl"),
            arch_data=arch_data,
        )

        dataloader = cls._get_data_loader(
            pth_to_pp_output,
            batch_size=train_data["batch_size"],
        )

        trainer = cls._get_trainer(model=model, data=dataloader, train_args=train_data)

        return cls(model=model, data=dataloader, trainer=trainer)

    @staticmethod
    def _get_model(
        model_name: str,
        feature_extractor: FeatureExtractors,
        label_encodec: LabelEncodec,
        arch_data: yamlDict,
    ) -> nn.Module:
        in_dim = feature_extractor.get_feature_dim()
        out_dim = label_encodec.get_class_dim()
        is_mhe = label_encodec.get_is_mhe()

        return Models[model_name](
            input_feature_dim=in_dim, class_dim=out_dim, mhe=is_mhe, **arch_data
        )

    @staticmethod
    def _get_data_loader(
        pp_path: Path,
        batch_size: int,
        num_workers: int = 0,
        gpu_available: bool = False,
    ):
        dataset = Hdf5Dataset(pp_path / "processed_data.h5", split=Split.TRAIN)
        sampler = SubsetRandomSampler(dataset.get_indices())
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_with_pack_pad_to_batch,
            num_workers=num_workers,
            pin_memory=gpu_available,
        )


    @staticmethod
    def _get_trainer(model:nn.Module, data:DataLoader, train_args:yamlDict) -> Trainer:
        return Trainer(max_epochs=train_args["max_epochs"], model=model, data=data, loss_fn=?, optimizer=?, save_pth=?)



# loss
# optis
# return trainer factory instead of trainer? -> partial
# save_pth to inject.