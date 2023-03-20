from __future__ import annotations

from pathlib import Path
from platform import architecture

import torch

from erinyes.data.loader import get_data_loader
from erinyes.models.classifier import PooledSeqClassifier
from erinyes.models.wav2vec_base import Wav2VecCTC
from erinyes.train.callbacks import TensorboardLoggingCallback
from erinyes.train.trainer import ObjectRecipe, Trainer
from erinyes.util.enums import Split
from sisyphus import tk


class W2V2TrainingJob(tk.Job):
    def __init__(
        self,
        data_path: tk.Path,
        pretrained_model_path: tk.Path,
        rqmts: dict | None = None,
        use_features=False,
        profile_first: bool = False,
    ) -> None:
        super().__init__()

        self.data_path = data_path
        self.pretrained_model_path = pretrained_model_path
        self.rqmts = rqmts if rqmts is not None else {"cpus": 1, "mem": 1, "time": 1}
        self.profile_first = profile_first

        self.use_features = use_features

        self.out_path = self.output_path("training")

        self.trainer = Trainer(
            max_epochs=200,
            loss_recipe=ObjectRecipe(
                name="cross_entropy", instance=torch.nn.CrossEntropyLoss
            ),
            optimizer_recipe=ObjectRecipe(
                name="adam_optimizer", instance=torch.optim.Adam
            ),
            save_pth=Path(self.out_path.get_path()),
            gpu_available=self.rqmts.get("gpu") is not None,
            callback_recipes=[
                ObjectRecipe(
                    name="tensorbaord",
                    instance=TensorboardLoggingCallback,
                    args={
                        "data_path": Path(self.data_path.get_path()),
                        "log_path": Path(self.out_path.get_path()),
                        "num_workers": self.rqtms.get("cpus",0),
                        "gpu_avalable": self.rqmts.get("gpu") is not None
                    },
                )
            ],
        )

    def get_model(self):
        label_encodec = torch.load(Path(self.data_path.get()) / "label_encodec.pt")

        return Wav2VecCTC(
            model_loc=self.pretrained_model_path.get_path(),
            frozen=self.use_features,
            return_conv_features=self.use_features,
            classifier=PooledSeqClassifier(
                out_dim=label_encodec.class_dim, hidden_dim=1024, is_mhe=False
            ),
        )

    def run(self):
        possible_state_loc = Path(self.out_path.get_path()) / "last"
        if possible_state_loc.exists():
            self.trainer = Trainer.from_state(possible_state_loc)
        else:
            model = self.get_model()
            train_data = get_data_loader(
                Path(self.data_path.get_path()),
                batch_size=4,
                split=Split.TRAIN,
                num_workers=self.rqmts["cpu"],
                gpu_available=self.rqmts.get("gpu") is not None,
            )
            self.trainer.prepare(model, train_data)

        self.trainer.fit()

    def run_profile(self):
        model = self.get_model()
        val_data = get_data_loader(
            Path(self.data_path.get_path()),
            batch_size=4,
            split=Split.VAL,
            num_workers=self.rqmts["cpu"],
            gpu_available=self.rqmts.get("gpu") is not None,
        )
        self.trainer.prepare(model, val_data)

        self.trainer.profile(log_path=self.out_path.get())

    def tasks(self):
        if self.profile_first:
            yield tk.Task("run_profile", rqmt=self.rqmts)
        yield tk.Task("run", rqmt=self.rqmts)
