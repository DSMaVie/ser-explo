from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Iterator

import yaml
from torch import nn
from torch.utils.data import DataLoader

from erinyes.data.util import get_data_loader
from erinyes.models import Models
from erinyes.train.other import LossFn, Optimizer
from erinyes.train.trainer import Trainer
from erinyes.util.enums import Split
from erinyes.util.env import Env
from erinyes.util.yaml import split_of, yamlDict

logger = logging.getLogger(__name__)


@dataclass
class TrainingsInstructions:
    model: nn.Module
    train_data: DataLoader
    val_data: DataLoader
    trainer_factory: Callable[..., Trainer]

    @classmethod
    def from_yaml(
        cls,
        pth_to_train_instructs: Path,
        pth_to_pp_output: Path,
        rqmts: dict,
        pth_to_pretrained_model: Path | None = None,
    ):
        logger.info("loading instructions from yaml")
        with pth_to_train_instructs.open("r") as train_file:
            train_data = yaml.safe_load(train_file)
            logger.info(f"found configuration {train_data}")

        pth_to_arch_params = (
            Env.load().INST_DIR / "architecture" / f"{train_data['architecture']}.yaml"
        )
        with pth_to_arch_params.open("r") as arch_file:
            arch_data = yaml.safe_load(arch_file)

        with (pth_to_pp_output / "feature_extractor.pkl").open("rb") as file:
            feature_extractor = pickle.load(file)
        with (pth_to_pp_output / "label_encoder.pkl").open("rb") as file:
            label_encodec = pickle.load(file)

        train_dataloader = get_data_loader(
            pth_to_pp_output,
            batch_size=train_data["batch_size"],
            split=Split.TRAIN,
            num_workers=rqmts.get("cpu", 0),
            gpu_available=True if rqmts.get("gpu") else False,
            pack=train_data.get("pack_sequences", True),
        )
        val_dataloader = get_data_loader(
            pth_to_pp_output,
            batch_size=train_data["batch_size"],
            split=Split.VAL,
            num_workers=rqmts.get("cpu", 0),
            gpu_available=True if rqmts.get("gpu") else False,
            pack=train_data.get("pack_sequences", True),
        )

        model = cls._get_model(
            arch_data,
            in_dim=feature_extractor.get_feature_dim(),
            out_dim=label_encodec.get_class_dim(),
            is_mhe=label_encodec.get_is_mhe(),
            model_loc=str(pth_to_pretrained_model),
        )
        logger.info("initialized model")

        trainer_factory = cls._get_trainer_factory(
            model.parameters(), train_args=train_data, is_mhe=label_encodec.get_is_mhe()
        )

        return cls(
            model=model,
            train_data=train_dataloader,
            val_data=val_dataloader,
            trainer_factory=trainer_factory,
        )

    @staticmethod
    def _get_model(
        arch_data: yamlDict,
        in_dim: int,
        out_dim: int,
        is_mhe: bool,
        model_loc: str | None = None,
    ) -> nn.Module:
        model_name, arch_data = split_of(arch_data, "model")
        model_inst = Models[model_name].value

        for key in arch_data:
            # resolve sub modules
            if isinstance(arch_data[key], dict):
                arch_data[key] = TrainingsInstructions._get_model(
                    arch_data[key], in_dim=in_dim, out_dim=out_dim, is_mhe=is_mhe
                )

            # resolve common dynamic variables
            if key == "out_dim":
                arch_data[key] = out_dim
            elif key == "in_dim":
                arch_data[key] = in_dim
            elif key == is_mhe:
                arch_data[key] = is_mhe
            elif key == "model_loc" and model_loc:
                arch_data[key] = model_loc

        return model_inst(**arch_data)

    @staticmethod
    def _get_trainer_factory(
        model_params: Iterator[nn.Parameter], train_args: yamlDict, is_mhe: bool = False
    ) -> Trainer:
        logger.info("building trainer factory")
        loss = LossFn[
            f"{'binary' if is_mhe else 'mc'}_{train_args['loss']}"
        ].value()  # pot. args in last bracket
        opti = Optimizer[train_args["optimizer"]].value(
            model_params
        )  # TODO: make args addable
        return partial(
            Trainer, max_epochs=train_args["max_epochs"], loss_fn=loss, optimizer=opti
        )
