import logging
from pathlib import Path

from recipe.infer import ClfInferenceJob
from recipe.preprocessing import PreprocessingJob
from recipe.train import TrainJob

from erinyes.util.env import Env
from sisyphus import tk

EXPERIMENT_NAME = "lstm_baseline"

logger = logging.getLogger(__name__)


def run_lstm_baseline():
    pp_inst_dir = Env.load().INST_DIR / "pp" / "mfcc"
    # pp_outputs = {}
    for pth in pp_inst_dir.rglob("*.yaml"):
        logger.info(f"Found instructions at {pth}. Starting PPLob for it.")
        pp_job = PreprocessingJob(tk.Path(str(pth)))
        tk.register_output(f"{EXPERIMENT_NAME}/pp/{pth.stem}", pp_job.out_pth)
    #     pp_outputs.update({pth.stem: pp_job.out_pth})

    # train_info = tk.Path(str(Env.load().INST_DIR / "train" / f"{EXPERIMENT_NAME}.yaml"))
    # for pp_name, pp_out_pth in pp_outputs.items():
    #     if pp_name != "iem_4emotions":
    #         continue
    #     train_job = TrainJob(
    #         pth_to_pp_output=pp_out_pth, pth_to_train_settings=train_info
    #     )

    #     tk.register_output(f"{EXPERIMENT_NAME}/training/{pp_name}", train_job.out_pth)

    #     inf_inst = tk.Path(
    #         str(Env.load().INST_DIR / "inference" / f"{EXPERIMENT_NAME}.yaml")
    #     )
    #     inf_job = InferenceJob(
    #         pth_to_data=pp_out_pth,
    #         pth_to_model_ckpts=train_job.out_pth,
    #         pth_to_inf_instructs=inf_inst,
    #     )

    #     tk.register_output(f"{EXPERIMENT_NAME}/results/{pp_name}", inf_job.out)

    # return inf_job.out
