import logging

from recipe.data_analysis import DataAnalysisJob
from recipe.download_pt_model import DownloadPretrainedModelJob
from recipe.preprocessing.ie4_w2v_clf import IEM4ProcessorForWav2Vec2
from recipe.preprocessing.rav_w2v_clf import RavdessW2VPreproJob

from erinyes.util.env import Env
from sisyphus import tk

# from recipe.train import TrainJob


logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "w2v_baselines"

DATA_CONDITIONS = {RavdessW2VPreproJob, IEM4ProcessorForWav2Vec2}
MODELS = ["w2v_pooled_clf"]


def run_w2v_baseline():
    model_dl_job = DownloadPretrainedModelJob(
        "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    )  # wav2vec2 xlsr ft on asr english (commonvoice)

    for data_pp_job in DATA_CONDITIONS:
        pp_job = data_pp_job()
        logger.info(
            f"Loading PPJob for data condition {pp_job.processor.name}. Starting Preprocessing for it."
        )
        tk.register_output(
            f"{EXPERIMENT_NAME}/{pp_job.processor.name}/data", pp_job.out_path
        )

        logger.info(f"loading analysis_job for {pp_job.processor.name}")
        pp_ana_job = DataAnalysisJob(pp_job.out_path, pp_job.label_column)
        tk.register_report(
            f"{EXPERIMENT_NAME}/{pp_job.processor.name}/data_stats.txt",
            pp_ana_job.stats,
        )

        # train_job = TrainJob(
        #     pth_to_pp_output=pp_job.out_path,
        #     pth_to_train_settings=train_info,
        #     pth_to_pretrained_model=model_dl_job.out,
        #     rqmts={"cpu": 2, "mem": 16, "gpu": 1, "time": 10},
        #     profile_first=False,
        # )

        # tk.register_output(f"{EXPERIMENT_NAME}/{pth.stem}/trained", train_job.out_path)
