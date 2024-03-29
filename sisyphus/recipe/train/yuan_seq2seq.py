from __future__ import annotations

import logging
import os
from functools import partial

from transformers import (
    AutoConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Wav2Vec2ForCTC,
)

from erinyes.data.hdf_dataset import Hdf5Dataset
from erinyes.data.loader import pad_collate
from erinyes.inference.metrics import BalancedEmotionErrorRate, EmotionErrorRate
from erinyes.train.callbacks import HFFreezeUnfreezeCallback

# from erinyes.inference.metrics_tracker import InTrainingsMetricsTracker
from erinyes.util.enums import Split
from sisyphus import Job, Task
from sisyphus import global_settings as gs
from sisyphus import tk

logger = logging.getLogger(__name__)


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
        self.rqmts = rqmts if rqmts is not None else {"cpu": 1, "mem": 1, "time": 1}
        self.profile_first = profile_first

        self.use_features = use_features

        self.out_path = self.output_path("training")
        self.model_args = self.output_var("model_args.pkl", pickle=True)
        self.model_class = self.output_var("model_class.pkl", pickle=True)

    def prepare_training(self):
        # label_encodec = torch.load(Path(self.data_path.get()) / "label_encodec.pt")
        # self.met_track = InTrainingsMetricsTracker(
        #     [EmotionErrorRate(), BalancedEmotionErrorRate(label_encodec.classes)]
        # )

        model_args = {}
        self.model_args.set(model_args)

        model_class = Wav2Vec2ForCTC
        self.model_class.set(model_class)

        return model_class.from_pretrained(
            self.pretrained_model_path.get_path(), **model_args
        )

    def run(self):
        train_args = Seq2SeqTrainingArguments(
            dataloader_num_workers=self.rqmts.get("cpu", 0),
            report_to="tensorboard",
            overwrite_output_dir=True,
            output_dir=self.out_path.get_path(),
            do_train=True,
            gradient_checkpointing=True,
            evaluation_strategy="steps",
            learning_rate=5e-5,
            max_steps=15_000,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=16,
            save_steps=50,
            logging_steps=25,
            eval_steps=50,
            weight_decay=0.005,
            warmup_steps=1000,
            # load_best_model_at_end=True,
            # metric_for_best_model="eval_loss",
        )

        data_path = gs.file_caching(self.data_path.join_right("processed_data.h5"))

        config = AutoConfig.from_pretrained(self.pretrained_model_path.get_path())
        model = self.prepare_training()
        train_data = Hdf5Dataset(
            src_path=data_path,
            split=Split.TRAIN,
        )
        eval_data = Hdf5Dataset(
            src_path=data_path,
            split=Split.VAL,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=train_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            # compute_metrics=self.met_track,
            callbacks=[HFFreezeUnfreezeCallback(10_000)],
            data_collator=partial(
                pad_collate,
                return_attention_mask=True,
                labels_are_seqs=True,
                padding_token_id=config.pad_token_id,
            ),
        )

        # Training
        if train_args.do_train:
            trainer.train()
            trainer.save_model()  # Saves the tokenizer too for easy upload
            trainer.save_state()

    def resume(self):
        self.train_args.resume_from_checkpoint = True
        self.run()

    def tasks(self):
        if self.profile_first:
            yield Task("run_profile", rqmt=self.rqmts)
        yield Task("run", resume="resume", rqmt=self.rqmts)
