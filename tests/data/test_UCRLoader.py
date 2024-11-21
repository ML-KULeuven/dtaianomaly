
import os
import pytest
import numpy as np
from dtaianomaly.data import UCRLoader, from_directory

UCR_DATA_PATH = 'data/UCR-time-series-anomaly-archive'
UCR_DATA_SET = '001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt'

data_available = pytest.mark.skipif(
    not os.path.isfile(f'{UCR_DATA_PATH}/{UCR_DATA_SET}'),
    reason='UCR file unavailable'
)
directory_available = pytest.mark.skipif(
    not os.path.isdir(UCR_DATA_PATH),
    reason='UCR directory unavailable'
)


@pytest.fixture
def loader():
    return UCRLoader(f'{UCR_DATA_PATH}/{UCR_DATA_SET}')


@pytest.fixture
def loaded(loader):
    return loader.load()


class TestUCRLoader:

    @data_available
    def test(self, loaded):
        assert loaded is not None

    @data_available
    def test_contains_anomaly(self, loaded):
        assert np.sum(loaded.y_test == 1) > 0

    @data_available
    def test_samples_match(self, loaded):
        assert loaded.X_test.shape[0] == loaded.y_test.shape[0]

    def test_faulty_path(self):
        with pytest.raises(FileNotFoundError):
            UCRLoader(path='bollocks')

    @data_available
    def test_str(self, loader):
        assert str(loader) == f"UCRLoader(path='{UCR_DATA_PATH}/{UCR_DATA_SET}')"

    @directory_available
    def test_from_directory(self):
        dataloaders = from_directory(UCR_DATA_PATH, UCRLoader)
        assert len(dataloaders) >= 1
        assert all([isinstance(loader, UCRLoader) for loader in dataloaders])
