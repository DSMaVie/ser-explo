# from config.lstm_baseline import run_lstm_baseline
from config.w2v import run_lj_baseline
from config.yuan_replication import run_yuan


def main():
    # run_lstm_baseline()
    run_lj_baseline()
    run_yuan()
