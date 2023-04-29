# from config.lstm_baseline import run_lstm_baseline
from config.lj_baselines import run_lj_fe_baseline, run_lj_ft_baseline
from config.yuan_replication import run_yuan


def main():
    # run_lstm_baseline()
    run_lj_ft_baseline()
    run_lj_fe_baseline()
