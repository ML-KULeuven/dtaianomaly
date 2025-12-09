import os

import numpy as np
import pytest

from dtaianomaly.data import TSBADLoader, from_directory

BASE_TSB_AD_DATA_PATH = "data/TSB-AD-"
TSB_AD_DATA_SET_UNIVARIATE = "001_NAB_id_1_Facility_tr_1007_1st_2014.csv"
TSB_AD_DATA_SET_MULTIVARIATE = "002_MSL_id_1_Sensor_tr_500_1st_900.csv"
TSB_AD_DATA_SET_WITH_TRAIN_LABELS = "009_NAB_id_9_Traffic_tr_500_1st_438.csv"

univariate_data_available = pytest.mark.skipif(
    not os.path.isfile(f"{BASE_TSB_AD_DATA_PATH}U/{TSB_AD_DATA_SET_UNIVARIATE}"),
    reason="File unavailable",
)
multivariate_data_available = pytest.mark.skipif(
    not os.path.isfile(f"{BASE_TSB_AD_DATA_PATH}M/{TSB_AD_DATA_SET_MULTIVARIATE}"),
    reason="File unavailable",
)
train_labels_data_available = pytest.mark.skipif(
    not os.path.isfile(f"{BASE_TSB_AD_DATA_PATH}U/{TSB_AD_DATA_SET_WITH_TRAIN_LABELS}"),
    reason="File unavailable",
)
univariate_directory_available = pytest.mark.skipif(
    not os.path.isdir(BASE_TSB_AD_DATA_PATH + "U"), reason="Directory unavailable"
)
multivariate_directory_available = pytest.mark.skipif(
    not os.path.isdir(BASE_TSB_AD_DATA_PATH + "M"), reason="Directory unavailable"
)


@pytest.fixture
def univariate_loader():
    return TSBADLoader(f"{BASE_TSB_AD_DATA_PATH}U/{TSB_AD_DATA_SET_UNIVARIATE}")


@pytest.fixture
def univariate_loaded(univariate_loader):
    return univariate_loader.load()


@pytest.fixture
def multivariate_loader():
    return TSBADLoader(f"{BASE_TSB_AD_DATA_PATH}M/{TSB_AD_DATA_SET_MULTIVARIATE}")


@pytest.fixture
def multivariate_loaded(multivariate_loader):
    return multivariate_loader.load()


@pytest.fixture
def supervised_loader():
    return TSBADLoader(f"{BASE_TSB_AD_DATA_PATH}U/{TSB_AD_DATA_SET_WITH_TRAIN_LABELS}")


@pytest.fixture
def supervised_loaded(supervised_loader):
    return supervised_loader.load()


class TestTSBADLoader:

    @univariate_data_available
    def test_univariate(self, univariate_loaded):
        assert univariate_loaded is not None
        assert np.sum(univariate_loaded.y_test == 1) > 0
        assert univariate_loaded.y_train is None
        assert univariate_loaded.X_test.shape[0] == univariate_loaded.y_test.shape[0]

    @multivariate_data_available
    def test_multivariate(self, multivariate_loaded):
        assert multivariate_loaded is not None
        assert np.sum(multivariate_loaded.y_test == 1) > 0
        assert multivariate_loaded.y_train is None
        assert (
            multivariate_loaded.X_test.shape[0] == multivariate_loaded.y_test.shape[0]
        )
        assert multivariate_loaded.X_test.shape[1] > 1

    @train_labels_data_available
    def test_supervised(self, supervised_loaded):
        assert supervised_loaded is not None
        assert np.sum(supervised_loaded.y_test == 1) > 0
        assert supervised_loaded.y_train is not None
        assert np.sum(supervised_loaded.y_train == 1) > 0
        assert supervised_loaded.X_test.shape[0] == supervised_loaded.y_test.shape[0]
        assert supervised_loaded.X_train.shape[0] == supervised_loaded.y_train.shape[0]

    @univariate_data_available
    def test_metadata_univariate(self, univariate_loader):
        assert univariate_loader.index == 1
        assert univariate_loader.dataset_name == "NAB"
        assert univariate_loader.id == 1
        assert univariate_loader.domain == "Facility"
        assert univariate_loader.train_index == 1007
        assert univariate_loader.first_anomaly == 2014

    @multivariate_data_available
    def test_metadata_multivariate(self, multivariate_loader):
        assert multivariate_loader.index == 2
        assert multivariate_loader.dataset_name == "MSL"
        assert multivariate_loader.id == 1
        assert multivariate_loader.domain == "Sensor"
        assert multivariate_loader.train_index == 500
        assert multivariate_loader.first_anomaly == 900

    def test_faulty_path(self):
        with pytest.raises(ValueError):
            TSBADLoader(path="bollocks")

    @univariate_directory_available
    def test_from_directory_univariate(self):
        dataloaders = from_directory(BASE_TSB_AD_DATA_PATH + "U", TSBADLoader)
        assert len(dataloaders) >= 1
        assert all([isinstance(loader, TSBADLoader) for loader in dataloaders])

    @multivariate_directory_available
    def test_from_directory_multivariate(self):
        dataloaders = from_directory(BASE_TSB_AD_DATA_PATH + "M", TSBADLoader)
        assert len(dataloaders) >= 1
        assert all([isinstance(loader, TSBADLoader) for loader in dataloaders])
