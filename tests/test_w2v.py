from erinyes.models.wav2vec_base import Wav2Vec


def test_w2v_webload():
    Wav2Vec.load_from_web()


def test_w2v_load():
    Wav2Vec()
