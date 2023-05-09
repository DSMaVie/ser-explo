# from config.lstm_baseline import run_lstm_baseline
from config.lj_baselines import run_lj_fe_baseline, run_lj_ft_baseline
from config.yuan_replication import run_yuan


def main():
    # run_lstm_baseline()
    run_lj_ft_baseline("facebook/wav2vec2-large")
    run_lj_fe_baseline("facebook/wav2vec2-large")
    run_lj_ft_baseline("facebook/wav2vec2-base-960h")
    run_lj_fe_baseline("facebook/wav2vec2-base-960h")
