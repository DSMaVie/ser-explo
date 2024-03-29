from __future__ import annotations

import logging
import os
from functools import partial
from pathlib import Path

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2Config,
    Wav2Vec2ForSequenceClassification,
)
from transformers.trainer_utils import get_last_checkpoint

from erinyes.data.hdf_dataset import Hdf5Dataset
from erinyes.data.loader import pad_collate
from erinyes.inference.metrics import BalancedEmotionErrorRate, EmotionErrorRate
from erinyes.inference.metrics_tracker import InTrainingsMetricsTracker
from erinyes.util.enums import Split
from sisyphus import Job, Task
from sisyphus import global_settings as gs
from sisyphus import tk

logger = logging.getLogger(__name__)


class LJFTTrainingJob(Job):
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
        self.rqmts = rqmts if rqmts is not None else {"cpu": 1, "mem": 1, "time": 1}
        self.profile_first = profile_first

        self.use_features = use_features

        self.out_path = self.output_path("training")
        self.model_args = self.output_var("model_args.pkl", pickle=True)
        self.model_class = self.output_var("model_class.pkl", pickle=True)

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

        model_args = {}
        self.model_args.set(model_args)

        model_class = Wav2Vec2ForSequenceClassification
        self.model_class.set(model_class)

        config = Wav2Vec2Config.from_pretrained(self.pretrained_model_path)
        config.num_labels = len(label_encodec.classes)
        config.classifier_proj_size = 1024

        return model_class.from_pretrained(
            self.pretrained_model_path.get_path(), config=config, **model_args
        )

    def run(self):
        train_args = TrainingArguments(
            output_dir=self.out_path.get_path(),
            do_train=True,
            num_train_epochs=20,
            gradient_checkpointing=True,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=16,
            save_steps=20,
            logging_steps=4,
            eval_steps=20,
            evaluation_strategy="steps",
            learning_rate=0.001,
            weight_decay=0.005,
            warmup_steps=50,
            dataloader_num_workers=self.rqmts.get("cpu", 0),
            report_to="tensorboard",
            overwrite_output_dir=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )
        model = self.prepare_training()
        data_path = gs.file_caching(self.data_path.join_right("processed_data.h5"))
        # breakpoint()

        # model.freeze_feature_encoder()
        train_data = Hdf5Dataset(
            src_path=data_path,
            split=Split.TRAIN,
        )
        eval_data = Hdf5Dataset(
            src_path=data_path,
            split=Split.VAL,
        )

        model.freeze_feature_encoder()
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            data_collator=partial(
                pad_collate, return_attention_mask=True, padding_token_id=0
            ),
            compute_metrics=self.met_track,
        )

        last_checkpoint = None
        logger.info(
            f" got out dir {train_args.output_dir}, do_train = {train_args.do_train}, overwrite = {train_args.overwrite_output_dir}, resume = {train_args.resume_from_checkpoint}, outdir content = {os.listdir(train_args.output_dir)}"
        )
        if (
            os.path.isdir(train_args.output_dir)
            and train_args.do_train
            and not train_args.overwrite_output_dir
        ):
            last_checkpoint = get_last_checkpoint(train_args.output_dir)
            if last_checkpoint is None and len(os.listdir(train_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({train_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif (
                last_checkpoint is not None
                and train_args.resume_from_checkpoint is None
            ):
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
        logger.info(f"last checkpoint is {last_checkpoint}")

        # Training
        if train_args.do_train:
            # first trainer pass

            # logger.info(f"found cp {checkpoint}")
            first_step_result = trainer.train()
            trainer.save_model()  # Saves the tokenizer too for easy upload
            trainer.save_state()



    def tasks(self):
        if self.profile_first:
            yield Task("run_profile", rqmt=self.rqmts)
        yield Task("run", rqmt=self.rqmts)
