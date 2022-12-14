import logging
from pathlib import Path

from erinyes.train.callbacks import (
    CombineCallbacks,
    ResetIfNoImprovement,
    TrackBestLoss,
)
from erinyes.train.instructions import TrainingsInstructions
from sisyphus import Job, Task, tk

logger = logging.getLogger(__name__)


class TrainJob(Job):
    def __init__(
        self, pth_to_pp_output: tk.Path, pth_to_train_settings: tk.Path
    ) -> None:
        super().__init__()

        self.pth_pp_output = Path(pth_to_pp_output)
        self.pth_to_train_settings = Path(pth_to_train_settings)
        self.out_pth = self.output_path("training")

    def run(self):
        instructions = TrainingsInstructions.from_yaml(
            self.pth_to_train_settings, pth_to_pp_output=self.pth_pp_output
        )

        model = instructions.model
        train_data = instructions.train_data
        val_data = instructions.val_data
        trainer = instructions.trainer_factory(
            model=model,
            train_data=train_data,
            save_pth=Path(self.out_pth),
            after_epoch=CombineCallbacks(
                callbacks=[
                    TrackBestLoss(val_data=val_data),
                    ResetIfNoImprovement(val_data=val_data),
                ]
            ),
        )

        trainer.fit()

    def tasks(self):
        yield Task("run")  # , rqmt={"engine": "krylov"})
