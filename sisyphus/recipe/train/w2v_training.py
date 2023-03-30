from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from transformers import (
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from functools import partial
from erinyes.data.hdf_dataset import Hdf5Dataset
from erinyes.data.loader import pad_collate
from erinyes.models.wav2vec_base import HFWav2VecCTCwithClf
from erinyes.util.enums import Split
from sisyphus import Job, Task, tk

logger = logging.getLogger(__name__)


class HFTrainingJob(Job):
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

        self.train_args = TrainingArguments(
            output_dir=self.out_path.get_path(),
            do_train=True,
            num_train_epochs=100,
            # gradient_checkpointing=True,
            save_strategy="epoch",
            dataloader_num_workers=self.rqmts.get("cpus", 0),
            report_to="tensorboard",
            overwrite_output_dir=True,
        )

    def get_model(self):
        label_encodec = torch.load(Path(self.data_path.get()) / "label_encodec.pt")

        return HFWav2VecCTCwithClf(
            model_loc=self.pretrained_model_path.get_path(),
            freeze_encoder=self.use_features,
            use_conv_features=self.use_features,
            clf_hidden_dim=1024,
            clf_out_dim=label_encodec.class_dim,
        )

    def run(self):
        model = self.get_model()
        train_data = Hdf5Dataset(
            src_path=self.data_path.get_path() + "/processed_data.h5",
            split=Split.TRAIN,
        )

        trainer = Trainer(
            model=model,
            args=self.train_args,
            train_dataset=train_data,
            data_collator=partial(pad_collate, return_dicts=True),
            # data_collator=DataCollatorWithPadding(
            #     tokenizer=Wav2Vec2CTCTokenizer.from_pretrained(
            #         self.pretrained_model_path.get_path()
            #     )
            # )
            # compute_metrics=compute_metrics,
        )

        last_checkpoint = None
        if (
            os.path.isdir(self.train_args.output_dir)
            and self.train_args.do_train
            and not self.train_args.overwrite_output_dir
        ):
            last_checkpoint = get_last_checkpoint(self.train_args.output_dir)
            if (
                last_checkpoint is None
                and len(os.listdir(self.train_args.output_dir)) > 0
            ):
                raise ValueError(
                    f"Output directory ({self.train_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif (
                last_checkpoint is not None
                and self.train_args.resume_from_checkpoint is None
            ):
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
        logger.info(f"last checkpoint is {last_checkpoint}")

        # Training
        if self.train_args.do_train:
            checkpoint = None
            if self.train_args.resume_from_checkpoint is not None:
                checkpoint = self.train_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint

            logger.info(f"found cp {checkpoint}")
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            # metrics = train_result.metrics

            # metrics["train_samples"] = len(train_data)

            trainer.save_model()  # Saves the tokenizer too for easy upload

            # trainer.log_metrics("train", metrics)
            # trainer.save_metrics("train", metrics)
            trainer.save_state()

    def resume(self):
        self.train_args.resume_from_checkpoint = True
        self.run()

    def tasks(self):
        if self.profile_first:
            yield Task("run_profile", rqmt=self.rqmts)
        yield Task("run", resume="resume", rqmt=self.rqmts)


# init -> train args
# _prep = init trainer get data and model
#       -> extra function?
# rewrite run
# model must use model output
# for that loss must be encoded in model!!
