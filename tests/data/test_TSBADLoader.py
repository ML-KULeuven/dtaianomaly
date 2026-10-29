import os

import numpy as np
import pandas as pd
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


def _write_tsb_ad_file(directory, file_name, n_channels=1):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(10, n_channels)))
    df["Label"] = [0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
    path = directory / file_name
    df.to_csv(path, index=False)
    return path


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

    @univariate_data_available
    def test_univariate_is_two_dimensional(self, univariate_loaded):
        assert univariate_loaded.X_test.ndim == 2
        assert univariate_loaded.X_train.ndim == 2

    def test_single_channel_keeps_two_dimensions(self, tmp_path):
        loaded = TSBADLoader(
            _write_tsb_ad_file(tmp_path, "001_NAB_id_1_Facility_tr_4_1st_6.csv")
        ).load()
        assert loaded.X_test.shape == (10, 1)
        assert loaded.X_train.shape == (4, 1)

    def test_multi_channel_keeps_all_channels(self, tmp_path):
        loaded = TSBADLoader(
            _write_tsb_ad_file(
                tmp_path, "002_MSL_id_1_Sensor_tr_4_1st_6.csv", n_channels=3
            )
        ).load()
        assert loaded.X_test.shape == (10, 3)
        assert loaded.X_train.shape == (4, 3)

    def test_test_set_is_full_series(self, tmp_path):
        loaded = TSBADLoader(
            _write_tsb_ad_file(tmp_path, "001_NAB_id_1_Facility_tr_4_1st_6.csv")
        ).load()
        assert loaded.X_test.shape[0] == 10
        assert np.array_equal(loaded.X_train, loaded.X_test[:4])

    def test_no_train_labels_if_first_anomaly_after_train_index(self, tmp_path):
        loaded = TSBADLoader(
            _write_tsb_ad_file(tmp_path, "001_NAB_id_1_Facility_tr_4_1st_6.csv")
        ).load()
        assert loaded.y_train is None

    def test_no_train_labels_if_first_anomaly_equals_train_index(self, tmp_path):
        loaded = TSBADLoader(
            _write_tsb_ad_file(tmp_path, "001_NAB_id_1_Facility_tr_6_1st_6.csv")
        ).load()
        assert loaded.y_train is None

    def test_train_labels_if_first_anomaly_before_train_index(self, tmp_path):
        loaded = TSBADLoader(
            _write_tsb_ad_file(tmp_path, "001_NAB_id_1_Facility_tr_8_1st_6.csv")
        ).load()
        assert np.array_equal(loaded.y_train, loaded.y_test[:8])

    @pytest.mark.parametrize(
        "file_name",
        [
            "foo.csv",
            "001_NAB_id_x_Facility_tr_10_1st_20.csv",
            "001_NAB_id_1_Facility_tr_10_1st_20.txt",
            "001_NAB_id_1_Facility_tr_10.csv",
            "001_NAB_Data_id_1_Facility_tr_10_1st_20.csv",
        ],
    )
    def test_invalid_file_name(self, tmp_path, file_name):
        path = _write_tsb_ad_file(tmp_path, file_name)
        with pytest.raises(ValueError, match="does not have the TSB-AD format"):
            TSBADLoader(path)

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
