import logging

from recipe.data_analysis import DataAnalysisJob
from recipe.download_pt_model import DownloadPretrainedModelJob
from recipe.preprocessing import PreprocessingJob

from erinyes.util.env import Env
from sisyphus import tk

# from recipe.train import TrainJob


EXPERIMENT_NAME = "w2v_baselines"

logger = logging.getLogger(__name__)


def run_w2v_baseline():
    pp_inst_dir = Env.load().INST_DIR / "pp" / "raw"
    train_info = tk.Path(
        str(Env.load().INST_DIR / "train" / f"{EXPERIMENT_NAME}.yaml"),
        hash_overwrite="train_info",
    )

    model_dl_job = DownloadPretrainedModelJob("facebook/wav2vec2-base-960h")
    for pth in pp_inst_dir.rglob("*.yaml"):
        logger.info(f"Found instructions at {pth}. Starting Preprocessing for it.")

        pp_info = tk.Path(str(pth))
        pp_job = PreprocessingJob(pp_info)
        tk.register_output(f"{EXPERIMENT_NAME}/{pth.stem}/processed", pp_job.out_pth)

        pp_ana_job = DataAnalysisJob(pp_info, pp_job.out_pth)
        tk.register_report(
            f"transcript_data_{pth.stem}", pp_ana_job.transcript_data
        )
        tk.register_report(f"priors_{pth.stem}", pp_ana_job.priors)

        #     pth_to_pp_output=pp_job.out_pth,
        #     pth_to_train_settings=train_info,
        #     pth_to_pretrained_model=model_dl_job.out,
        #     rqmts={"cpu": 4, "mem": 48, "gpu": 1, "time": 10},
        #     profile_first=False,
        # )

        # tk.register_output(f"{EXPERIMENT_NAME}/{pth.stem}/trained", train_job.out_pth)
