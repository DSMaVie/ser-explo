from enum import Enum
from erinyes.models.classifier import PooledSeqClassifier

from erinyes.models.wav2vec_base import Wav2Vec, Wav2VecCTC

from .SoftAttnLstm import SoftAttnLstmClf


class Models(Enum):
    softattn_lstm_clf = SoftAttnLstmClf
    w2v = Wav2Vec
    w2v_ctc = Wav2VecCTC
    pooled_seq_clf = PooledSeqClassifier
