import logging
from functools import partial

from recipe.data_analysis import DataAnalysisJob
from recipe.decide import ArgMaxDecision
from recipe.download_pt_model import DownloadPretrainedModelJob
from recipe.infer import ClfInferenceJob
from recipe.preprocessing.ie4_w2v_clf import (
    IEM4ProcessorForWav2Vec2,
    IEM4ProcessorForWav2Vec2WithModelFeatures,
)
from recipe.preprocessing.rav_w2v_clf import (
    RavdessW2VPreproJob,
    RavdessW2VPreproJobWithModelFeatures,
)
from recipe.train.lj_fe import LJFETrainingJob
from recipe.train.lj_ft import LJFTTrainingJob

from sisyphus import tk

logger = logging.getLogger(__name__)


def run_lj_ft_baseline(base_model: str):
    model_dl_job = DownloadPretrainedModelJob(
        base_model,
        just_model=True,
        rqmts={"cpu": 1, "mem": 10, "gpu": 0, "time": 1},
    )  # wav2vec2 xlsr ft on asr english (commonvoice)

    pp_jobs = [RavdessW2VPreproJob, IEM4ProcessorForWav2Vec2]

    for data_pp_job in pp_jobs:
        pp_job = data_pp_job()
        logger.info(
            f"Loading PPJob for data condition {pp_job.processor.name}. Starting Preprocessing for it."
        )
        # tk.register_output(f"pp/{pp_job.processor.name}/data", pp_job.out_path)

        logger.info(f"loading analysis_job for {pp_job.processor.name}")
        pp_ana_job = DataAnalysisJob(pp_job.out_path, pp_job.label_column)
        tk.register_report(
            f"pp/{pp_job.processor.name}/data_stats.txt",
            pp_ana_job.stats,
        )

        train_job = LJFTTrainingJob(
            data_path=pp_job.out_path,
            pretrained_model_path=model_dl_job.out,
            rqmts={"cpu": 4, "mem": 36, "gpu": 1, "time": 72},
            profile_first=False,
        )
        # tk.register_output(f"/{pp_job.processor.name}/{train_desc}", train_job.out_path)

        infer_job = ClfInferenceJob(
            path_to_model_ckpts=train_job.out_path,
            path_to_data=pp_job.out_path,
            model_args=train_job.model_args,
            model_class=train_job.model_class,
            rqmts={"cpu": 1, "mem": 16, "gpu": 1, "time": 1},
        )

        dec_job = ArgMaxDecision(
            path_to_inferences=infer_job.pred_out, class_labels=infer_job.class_labels
        )
        tk.register_output(
            f"{pp_job.processor.name}/lj_finetune/decisions", dec_job.decisions
        )
        tk.register_report(
            f"{pp_job.processor.name}/lj_finetune/results", dec_job.result
        )


def run_lj_fe_baseline(base_model: str):
    model_dl_job = DownloadPretrainedModelJob(
        base_model,
        rqmts={"cpu": 1, "mem": 10, "gpu": 0, "time": 1},
        just_model=True,
    )  # wav2vec2 xlsr ft on asr english (commonvoice)

    pp_jobs = [
        partial(
            RavdessW2VPreproJobWithModelFeatures, rqmts={"cpu": 2, "mem": 10, "time": 4}
        ),
        partial(
            IEM4ProcessorForWav2Vec2WithModelFeatures,
            rqmts={"cpu": 2, "mem": 10, "time": 4},
        ),
    ]

    for data_pp_job in pp_jobs:
        pp_job = data_pp_job(path_to_tokenizer=model_dl_job.out)
        logger.info(
            f"Loading PPJob for data condition {pp_job.processor.name}. Starting Preprocessing for it."
        )
        # tk.register_output(f"pp/{pp_job.processor.name}/data", pp_job.out_path)

        logger.info(f"loading analysis_job for {pp_job.processor.name}")
        pp_ana_job = DataAnalysisJob(pp_job.out_path, pp_job.label_column)
        tk.register_report(
            f"pp/{pp_job.processor.name}/data_stats.txt",
            pp_ana_job.stats,
        )

        train_job = LJFETrainingJob(
            data_path=pp_job.out_path,
            pretrained_model_path=model_dl_job.out,
            rqmts={"cpu": 4, "mem": 20, "gpu": 1, "time": 8},
            profile_first=False,
        )
        # tk.register_output(f"/{pp_job.processor.name}/{train_desc}", train_job.out_path)

        infer_job = ClfInferenceJob(
            path_to_model_ckpts=train_job.out_path,
            path_to_data=pp_job.out_path,
            model_args=train_job.model_args,
            model_class=train_job.model_class,
            rqmts={"cpu": 1, "mem": 16, "gpu": 1, "time": 1},
        )

        dec_job = ArgMaxDecision(
            path_to_inferences=infer_job.pred_out, class_labels=infer_job.class_labels
        )
        tk.register_output(
            f"{pp_job.processor.name}/lj_ft/decisions", dec_job.decisions
        )
        tk.register_report(f"{pp_job.processor.name}/lj_ft/results", dec_job.result)
