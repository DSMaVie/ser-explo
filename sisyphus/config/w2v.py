import itertools
import logging
from functools import partial

from recipe.data_analysis import DataAnalysisJob
from recipe.decide import ArgMaxDecision
from recipe.download_pt_model import DownloadPretrainedModelJob
from recipe.infer import InferenceJob
from recipe.preprocessing.ie4_w2v_clf import IEM4ProcessorForWav2Vec2
from recipe.preprocessing.rav_w2v_clf import RavdessW2VPreproJob
from recipe.train.w2v_training import HFTrainingJob

from sisyphus import tk

logger = logging.getLogger(__name__)

DATA_CONDITIONS = [RavdessW2VPreproJob, IEM4ProcessorForWav2Vec2]
TRAIN_CONDITIONS = {
    "lj_finetune": partial(HFTrainingJob, use_features=False),
    "lj_feature_extraction": partial(HFTrainingJob, use_features=True),
}


def run_lj_baseline():
    model_dl_job = DownloadPretrainedModelJob(
        "jonatasgrosman/wav2vec2-large-xlsr-53-english",
        rqmts={"cpu": 1, "mem": 10, "gpu": 0, "time": 1},
    )  # wav2vec2 xlsr ft on asr english (commonvoice)

    for data_pp_job, (train_desc, train_job) in itertools.product(
        DATA_CONDITIONS, TRAIN_CONDITIONS.items()
    ):
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

        train_job = train_job(
            data_path=pp_job.out_path,
            pretrained_model_path=model_dl_job.out,
            rqmts={"cpu": 4, "mem": 16, "gpu": 1, "time": 24},
            profile_first=False,
        )
        # tk.register_output(f"/{pp_job.processor.name}/{train_desc}", train_job.out_path)

        infer_job = InferenceJob(
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
            f"/{pp_job.processor.name}/{train_desc}/decisions", dec_job.decisions
        )
        tk.register_report(
            f"/{pp_job.processor.name}/{train_desc}/results", dec_job.result
        )
