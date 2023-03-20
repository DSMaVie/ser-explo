from __future__ import annotations

import logging
import torch
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Iterator

import yaml
from torch import nn
from torch.utils.data import DataLoader

from erinyes.data.loader import get_data_loader
from erinyes.models import Models
from erinyes.train.other import LossFn, Optimizer
from erinyes.train.trainer import Trainer
from erinyes.util.enums import Split
from erinyes.util.env import Env
from erinyes.util.yaml import split_of, yamlDict

logger = logging.getLogger(__name__)


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


    train_dataloader = get_data_loader(
        pth_to_pp_output,
        batch_size=train_data["batch_size"],
        split=Split.TRAIN,
        num_workers=rqmts.get("cpu", 0),
        gpu_available=True if rqmts.get("gpu") else False,
    )

    val_dataloader = get_data_loader(
        pth_to_pp_output,
        batch_size=train_data["batch_size"],
        split=Split.VAL,
        num_workers=rqmts.get("cpu", 0),
        gpu_available=True if rqmts.get("gpu") else False,
    )
    feature_extractor = torch.load(pth_to_pp_output / "feature_extractor.pt")
    label_encodec = torch.load(pth_to_pp_output / "label_encodec.pt")

    model = cls._get_model(
        arch_data,
        in_dim=feature_extractor.get_feature_dim(),
        out_dim=label_encodec.class_dim,
        is_mhe=label_encodec.is_mhe,
        model_loc=str(pth_to_pretrained_model),
    )




@staticmethod
def _get_model(
    path_to_pp_artifacts: Path,
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
            arch_data[key] = _get_model(
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

    logger.info("initialized model")
    return model_inst(**arch_data)
