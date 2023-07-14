from __future__ import annotations

import logging

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from erinyes.data.loader import get_data_loader
from erinyes.train.trainer import Trainer
from erinyes.util.enums import Split
from erinyes.util.env import return_gpu_utilization

logger = logging.getLogger(__name__)


class SaveBestLoss:
    def __init__(self, val_data: DataLoader) -> None:
        self.data = val_data
        self.best_loss = -np.inf

    def after_batch(self, _):
        return

    def after_epoch(self, train_state: Trainer):
        train_state.model.eval()
        loss = -np.inf
        for batch_idx, (x, y) in tqdm(
            enumerate(self.data),
            desc="Validation: Current Batch in Epoch",
        ):
            x = x.to(train_state._train_device)
            y = y.to(train_state._train_device)

            # calc model output and loss
            pred = train_state.model(x)
            loss += train_state.loss(pred, y.long())

        avg_loss = loss / batch_idx
        if avg_loss > self.best_loss:
            self.best_loss = avg_loss
            train_state.save_state(train_state.save_pth / "best_val_loss")


class ResetStateIfNoImprovement:
    def __init__(self, val_data: DataLoader) -> None:
        self.data = val_data
        self.best_loss = -np.inf
        self.best_model = None

    def after_batch(self, _):
        return

    def after_epoch(self, train_state: Trainer):
        train_state.model.eval()
        loss = -np.inf
        for batch_idx, (x, y) in tqdm(
            enumerate(self.data),
            desc="Validation: Current Batch in Epoch",
        ):
            x = x.to(train_state._train_device)
            y = y.to(train_state._train_device)

            # calc model output and loss
            pred = train_state.model(x)
            loss += train_state.loss_fn(pred, y.long())

        avg_loss = loss / batch_idx

        if (
            avg_loss < self.best_loss and self.best_model
        ):  # only do this of best model exists
            logger.info(
                f"Overwriting batch info because best_loss so far is {self.best_loss}, while this time we got an avg validation loss of {avg_loss}."
            )
            train_state.completed_batches -= 1
            train_state.completed_epochs -= 1
            train_state.model = self.best_model
        else:
            self.best_model = train_state.model
            self.best_loss = avg_loss


class TensorboardLoggingCallback:  # TODO add metrics
    def __init__(
        self,
        data_path: DataLoader,
        log_path: str,
        batch_size=64,
        num_workers=0,
        gpu_available=False,
    ) -> None:
        self.writer = SummaryWriter(log_dir=log_path)
        self.data = get_data_loader(
            data_path,
            batch_size=batch_size,
            split=Split.VAL,
            num_workers=num_workers,
            gpu_available=gpu_available,
        )

        self.idx = 0

    def after_batch(self, train_state: Trainer):
        self.writer.add_scalar(
            "loss/train",
            train_state.current_loss,
            self.idx,
        )
        self.idx += 1

    def after_epoch(self, train_state: Trainer) -> None:
        train_state.model.eval()
        loss = -np.inf
        for batch_idx, (x, y) in tqdm(
            enumerate(self.data),
            desc="Validation: Current Batch in Epoch",
        ):
            x = x.to(train_state._train_device)
            y = y.to(train_state._train_device)

            # calc model output and loss
            pred = train_state.model(x)
            loss += train_state.loss(pred, y)

        avg_loss = loss / batch_idx
        self.writer.add_scalar(
            "loss/train",
            train_state.current_loss,
            self.idx,
        )
        self.writer.add_scalar("loss/val", avg_loss, self.idx)

    def __del__(self):
        self.writer.close()


class TrackVRAMUsage:
    def after_batch(self, _):
        return

    def after_step(self, _):
        gpu_ut = return_gpu_utilization()
        logger.info(f"{gpu_ut:.2f}MB currently used on VRAM.")


class HFFreezeUnfreezeCallback(TrainerCallback):
    def __init__(self, unfreeze_after_step: int):
        self.trigger_step = unfreeze_after_step
        self.has_been_unfrozen = False

    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        for param in kwargs["model"].wav2vec2.parameters():
            param.requires_grad = False

        logger.info("Base Model has been frozen for training at start")

        return super().on_init_end(args, state, control, **kwargs)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not self.has_been_unfrozen:
            if state.global_step >= self.trigger_step:
                for param in kwargs["model"].wav2vec2.parameters():
                    param.requires_grad = True

                kwargs["model"].freeze_feature_encoder()
                self.has_been_unfrozen = True
                logger.info("base model has been unfrozen!")

        return super().on_step_end(args, state, control, **kwargs)
