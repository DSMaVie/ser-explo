import psutil

from erinyes.data.hdf_dataset import Hdf5Dataset
from erinyes.util.enums import Split


def test_hdf5_mem():
    dataset = Hdf5Dataset(
        "/work/smt4/thulke/vieweg/SER/Code/sisyphus/work/preprocessing/PreprocessingJob.pT3JaqtGy5YR/output/processed_data.h5",
        split=Split.TRAIN,
    )

    for i, idx in enumerate(dataset.available_indices):
        bathc = dataset[idx]
        print(
            f"at iter {i} we have {psutil.virtual_memory()[3]/1000000000} GB ram usage"
        )
