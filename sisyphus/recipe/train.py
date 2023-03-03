from __future__ import annotations

import logging
from pathlib import Path

from erinyes.train.callbacks import (ResetStateIfNoImprovement, SaveBestLoss,
                                     TensorboardLoggingCallback)
from erinyes.train.instructions import TrainingsInstructions
from sisyphus import Job, Task, tk

logger = logging.getLogger(__name__)


class TrainJob(Job):
    def __init__(
        self,
        pth_to_pp_output: tk.Path,
        pth_to_train_settings: tk.Path,
        rqmts: dict,
        pth_to_pretrained_model: tk.Path | None = None,
        profile:bool=True
    ) -> None:
        super().__init__()
        logger.info("starting trainjob.")

        self.pth_pp_output = Path(pth_to_pp_output)
        self.pth_to_train_settings = Path(pth_to_train_settings)
        self.pth_to_pretrained_model = pth_to_pretrained_model

        self.rqmts = rqmts
        self.profile = profile

        self.out_pth = self.output_path("training")

    def run(self):
        instructions = TrainingsInstructions.from_yaml(
            self.pth_to_train_settings,
            rqmts=self.rqmts,
            pth_to_pp_output=self.pth_pp_output,
            pth_to_pretrained_model=self.pth_to_pretrained_model,
        )

        model = instructions.model
        train_data = instructions.train_data
        val_data = instructions.val_data
        trainer = instructions.trainer_factory(
            model=model,
            train_data=train_data,
            save_pth=Path(self.out_pth),
            callbacks=[
                SaveBestLoss(val_data=val_data),
                ResetStateIfNoImprovement(val_data=val_data),
                TensorboardLoggingCallback(val_data=val_data, log_pth=self.out_pth),
            ],
        )

        trainer.fit()

    def profile(self):
        instructions = TrainingsInstructions.from_yaml(
            self.pth_to_train_settings,
            rqmts=self.rqmts,
            pth_to_pp_output=self.pth_pp_output,
            pth_to_pretrained_model=self.pth_to_pretrained_model,
        )

        model = instructions.model
        train_data = instructions.train_data
        val_data = instructions.val_data
        trainer = instructions.trainer_factory(
            model=model,
            train_data=train_data,
            save_pth=Path(self.out_pth),
        )

        trainer.profile(val_data, log_path=self.out_pth)

    def tasks(self):
        if self.profile:
            yield Task("profile", rqmt=self.rqmts)
        yield Task("run", rqmt=self.rqmts)
