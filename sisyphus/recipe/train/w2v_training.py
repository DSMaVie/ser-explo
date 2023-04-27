from __future__ import annotations

import logging
import os
from functools import partial
from pathlib import Path

import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
    Wav2Vec2ForCTC,
)
from transformers.trainer_utils import get_last_checkpoint

from erinyes.data.hdf_dataset import Hdf5Dataset
from erinyes.data.loader import pad_collate
from erinyes.inference.metrics import BalancedEmotionErrorRate, EmotionErrorRate
from erinyes.inference.metrics_tracker import InTrainingsMetricsTracker
from erinyes.models.wav2vec_base import HFWav2Vec2withClf
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
        self.model_args = self.output_var("model_args.pkl", pickle=True)
        self.model_class = self.output_var("model_class.pkl", pickle=True)

        self.train_args = TrainingArguments(
            output_dir=self.out_path.get_path(),
            do_train=True,
            num_train_epochs=25,
            gradient_checkpointing=True,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            save_steps=10,
            logging_steps=10,
            eval_steps=10,
            evaluation_strategy="steps",
            learning_rate=0.001,
            weight_decay=0.005,
            warmup_steps=100,
            dataloader_num_workers=self.rqmts.get("cpus", 0),
            report_to="tensorboard",
            overwrite_output_dir=True,
        )

    def prepare_training(self):
        label_encodec = torch.load(Path(self.data_path.get()) / "label_encodec.pt")
        self.met_track = InTrainingsMetricsTracker(
            [
                EmotionErrorRate(),
                BalancedEmotionErrorRate(
                    label_encodec.classes, return_per_emotion=True
                ),
            ]
        )

        model_args = {
            "freeze_encoder": self.use_features,
            "use_conv_features": self.use_features,
            "clf_hidden_dim": 512 if self.use_features else 1024,
            "clf_out_dim": label_encodec.class_dim,
        }
        self.model_args.set(model_args)

        model_class = HFWav2Vec2withClf
        self.model_class.set(model_class)

        return model_class.from_pretrained(
            self.pretrained_model_path.get_path(), **model_args
        )

    def run(self):
        model = self.prepare_training()
        train_data = Hdf5Dataset(
            src_path=self.data_path.get_path() + "/processed_data.h5",
            split=Split.TRAIN,
        )
        eval_data = Hdf5Dataset(
            src_path=self.data_path.get_path() + "/processed_data.h5",
            split=Split.VAL,
        )

        trainer = Trainer(
            model=model,
            args=self.train_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            data_collator=partial(pad_collate, return_attention_mask=False),
            compute_metrics=self.met_track,
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


class HFSeq2SeqTrainingJob(Job):
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
        self.model_args = self.output_var("model_args.pkl", pickle=True)
        self.model_class = self.output_var("model_class.pkl", pickle=True)

        self.train_args = Seq2SeqTrainingArguments(
            output_dir=self.out_path.get_path(),
            do_train=True,
            num_train_epochs=15,
            gradient_checkpointing=True,
            save_steps=100,
            logging_steps=100,
            learning_rate=0.5e-5,
            weight_decay=0.005,
            warmup_steps=250,
            dataloader_num_workers=self.rqmts.get("cpus", 0),
            report_to="tensorboard",
            overwrite_output_dir=True,
        )

    def prepare_training(self):
        label_encodec = torch.load(Path(self.data_path.get()) / "label_encodec.pt")
        # self.met_track = InTrainingsMetricsTracker([
        #     EmotionErrorRate(),
        #     BalancedEmotionErrorRate(label_encodec.classes)
        # ])

        model_args = {
            "freeze_encoder": self.use_features,
            "use_conv_features": self.use_features,
            "clf_hidden_dim": 512 if self.use_features else 1024,
            "clf_out_dim": label_encodec.class_dim,
        }
        self.model_args.set(model_args)

        model_class = Wav2Vec2ForCTC
        self.model_class.set(model_class)

        return model_class.from_pretrained(
            self.pretrained_model_path.get_path(), **model_args
        )

    def run(self):
        model = self.prepare_training()
        train_data = Hdf5Dataset(
            src_path=self.data_path.get_path() + "/processed_data.h5",
            split=Split.TRAIN,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=self.train_args,
            train_dataset=train_data,
            data_collator=partial(pad_collate, return_dicts=True, pad_both=True),
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
