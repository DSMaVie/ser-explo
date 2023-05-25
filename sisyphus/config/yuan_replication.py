import logging

from recipe.data_analysis import DataAnalysisJob
from recipe.decide.sequence_level import ArgMaxSeqDecision
from recipe.download_pt_model import DownloadPretrainedModelWithPhonemeTokenzier
from recipe.infer import Seq2SeqInferenceJob
from recipe.preprocessing.ie4_w2v_clf import IEM4ProcessorForWav2Vec2WithPhonemes
from recipe.preprocessing.rav_w2v_clf import RavdessW2VPreproJobWithPhonemes
from recipe.train.yuan_seq2seq import HFSeq2SeqTrainingJob
from recipe.train_analysis import TrainAnalysisJob

from sisyphus import tk

logger = logging.getLogger(__name__)

# TRAIN_CONDITIONS = {
#     "lj_finetune": partial(HFTrainingJob, use_features=False),
#     "lj_feature_extraction": partial(HFTrainingJob, use_features=True),
# }


def run_yuan():
    model_dl_job = DownloadPretrainedModelWithPhonemeTokenzier(
        "facebook/wav2vec2-base-960h",
        rqmts={"cpu": 1, "mem": 10, "gpu": 0, "time": 1},
    )  # wav2vec2 base model and custom tok

    for data_pp_job in [
        RavdessW2VPreproJobWithPhonemes,
        IEM4ProcessorForWav2Vec2WithPhonemes,
    ]:
        data_job = data_pp_job(model_dl_job.out)

        tk.register_output(f"pp/{data_job.processor.name}/data", data_job.out_path)

        pp_job = data_pp_job(path_to_tokenizer=model_dl_job.out)
        logger.info(
            f"Loading PPJob for data condition {pp_job.processor.name}. Starting Preprocessing for it."
        )

        logger.info(f"loading analysis_job for {pp_job.processor.name}")
        pp_ana_job = DataAnalysisJob(pp_job.out_path, pp_job.label_column)
        tk.register_report(
            f"pp/{pp_job.processor.name}/data_stats.txt",
            pp_ana_job.stats,
        )

        train_job = HFSeq2SeqTrainingJob(
            data_path=pp_job.out_path,
            pretrained_model_path=pp_job.new_model_loc,
            rqmts={"cpu": 4, "mem": 16, "gpu": 1, "time": 24},
            profile_first=False,
        )
        # tk.register_output(f"{pp_job.processor.name}/yuan_base", train_job.out_path)

        train_ana_job = TrainAnalysisJob(train_job.out_path, train_job.model_class)
        tk.register_output(
            f"{pp_job.processor.name}/yuan/train_analysis", train_ana_job.out
        )

        infer_job = Seq2SeqInferenceJob(
            path_to_model_ckpts=train_job.out_path,
            path_to_data=pp_job.out_path,
            model_args=train_job.model_args,
            model_class=train_job.model_class,
            rqmts={"cpu": 1, "mem": 16, "gpu": 1, "time": 1},
        )

        dec_job = ArgMaxSeqDecision(
            path_to_inferences=infer_job.pred_out, class_labels=infer_job.class_labels
        )
        tk.register_output(
            f"/{pp_job.processor.name}/yuan/decisions", dec_job.decisions
        )
        tk.register_report(f"/{pp_job.processor.name}/yuan/results", dec_job.result)
