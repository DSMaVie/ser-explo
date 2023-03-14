from __future__ import annotations

import logging

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from erinyes.train.trainer import Trainer
from erinyes.util.env import print_gpu_utilization

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
            loss += train_state.loss_fn(pred, y.long())

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
    def __init__(self, val_data: DataLoader, log_pth: str) -> None:
        self.writer = SummaryWriter(log_dir=log_pth)
        self.val_data = val_data

    def after_batch(self, train_state: Trainer):
        self.writer.add_scalar(
            "loss/train",
            train_state.current_loss,
            train_state.completed_epochs * train_state.max_epochs
            + train_state.completed_batches,
        )

    def after_epoch(self, train_state: Trainer) -> None:
        train_state.model.eval()
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
        self.writer.add_scalar(
            "loss/train",
            train_state.current_loss,
            train_state.completed_epochs * train_state.max_epochs,
        )
        self.writer.add_scalar(
            "loss/val", avg_loss, train_state.completed_epochs * train_state.max_epochs
        )

    def __del__(self):
        self.writer.close()


class TrackVRAMUsage():
    def after_batch(self, _):
        print_gpu_utilization()

    def after_step(self, _):
        return