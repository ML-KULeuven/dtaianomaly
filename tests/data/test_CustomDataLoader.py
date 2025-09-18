
import os

import pandas as pd
import pytest

from dtaianomaly.data import CustomDataLoader
from dtaianomaly.anomaly_detection import Supervision


def write_unsupervised_data(path):
    with open(path, 'w') as f:
        f.write("time,label,f1,f2,f3\n")
        for i in range(10):
            label = 1 if 3 <= i <= 5 else 0
            f.write(f"{i},{label},{i+5},{i*10},{i*10+5}\n")


def write_semi_supervised_data(path_test, path_train):
    write_unsupervised_data(path_test)
    with open(path_train, 'w') as f:
        f.write("time,f1,f2,f3\n")
        for i in range(10):
            f.write(f"{i},{i+5},{i*10},{i*10+5}\n")


def write_semi_supervised_data_zero_labels(path_test, path_train):
    write_unsupervised_data(path_test)
    with open(path_train, 'w') as f:
        f.write("time,label,f1,f2,f3\n")
        for i in range(10):
            label = 0
            f.write(f"{i},{label},{i+5},{i*10},{i*10+5}\n")


def write_supervised_data(path_test, path_train):
    write_unsupervised_data(path_test)
    write_unsupervised_data(path_train)


def drop_time(path):
    pd.read_csv(path).drop(columns="time").to_csv(path, index=False)


class TestCustomDataLoader:

    @pytest.mark.parametrize("time", [True, False])
    def test_unsupervised(self, time, tmp_path):
        write_unsupervised_data(tmp_path / 'unsupervised.csv')
        if not time:
            drop_time(tmp_path / 'unsupervised.csv')
        dataset = CustomDataLoader(tmp_path / 'unsupervised.csv').load()
        compatible_supervision = dataset.compatible_supervision()
        assert Supervision.UNSUPERVISED in compatible_supervision
        assert Supervision.SEMI_SUPERVISED not in compatible_supervision
        assert Supervision.SUPERVISED not in compatible_supervision

    @pytest.mark.parametrize("time_train", [True, False])
    @pytest.mark.parametrize("time_test", [True, False])
    def test_semi_supervised(self, time_train, time_test, tmp_path):
        write_semi_supervised_data(tmp_path / 'semi_supervised.csv', tmp_path / 'semi_supervised_train.csv')
        if not time_train:
            drop_time(tmp_path / 'semi_supervised_train.csv')
        if not time_test:
            drop_time(tmp_path / 'semi_supervised.csv')
        dataset = CustomDataLoader(tmp_path / 'semi_supervised.csv', tmp_path / 'semi_supervised_train.csv').load()
        compatible_supervision = dataset.compatible_supervision()
        assert Supervision.UNSUPERVISED in compatible_supervision
        assert Supervision.SEMI_SUPERVISED in compatible_supervision
        assert Supervision.SUPERVISED not in compatible_supervision

    @pytest.mark.parametrize("time_train", [True, False])
    @pytest.mark.parametrize("time_test", [True, False])
    def test_semi_supervised_zero_labels(self, time_train, time_test, tmp_path):
        write_semi_supervised_data_zero_labels(tmp_path / 'semi_supervised.csv', tmp_path / 'semi_supervised_train.csv')
        if not time_train:
            drop_time(tmp_path / 'semi_supervised_train.csv')
        if not time_test:
            drop_time(tmp_path / 'semi_supervised.csv')
        dataset = CustomDataLoader(tmp_path / 'semi_supervised.csv', tmp_path / 'semi_supervised_train.csv').load()
        compatible_supervision = dataset.compatible_supervision()
        assert Supervision.UNSUPERVISED in compatible_supervision
        assert Supervision.SEMI_SUPERVISED in compatible_supervision
        assert Supervision.SUPERVISED not in compatible_supervision

    @pytest.mark.parametrize("time_train", [True, False])
    @pytest.mark.parametrize("time_test", [True, False])
    def test_supervised(self, time_train, time_test, tmp_path):
        write_supervised_data(tmp_path / 'supervised.csv', tmp_path / 'supervised_train.csv')
        if not time_train:
            drop_time(tmp_path / 'supervised_train.csv')
        if not time_test:
            drop_time(tmp_path / 'supervised.csv')
        dataset = CustomDataLoader(tmp_path / 'supervised.csv', tmp_path / 'supervised_train.csv').load()
        compatible_supervision = dataset.compatible_supervision()
        assert Supervision.UNSUPERVISED in compatible_supervision
        assert Supervision.SEMI_SUPERVISED in compatible_supervision
        assert Supervision.SUPERVISED in compatible_supervision

    def test_non_existing_test_path(self):
        with pytest.raises(FileNotFoundError):
            CustomDataLoader("invalid-test-path.csv")

    def test_non_existing_train_path(self, tmp_path):
        write_unsupervised_data(tmp_path / 'unsupervised.csv')
        with pytest.raises(FileNotFoundError):
            CustomDataLoader(tmp_path / 'unsupervised.csv', 'invalid-train-path.csv')

    def test_none_test_path(self):
        with pytest.raises(FileNotFoundError):
            CustomDataLoader(None)

    def test_str(self, tmp_path):
        write_unsupervised_data(tmp_path / 'unsupervised.csv')
        write_semi_supervised_data(tmp_path / 'semi_supervised.csv', tmp_path / 'semi_supervised_train.csv')
        write_supervised_data(tmp_path / 'supervised.csv', tmp_path / 'supervised_train.csv')

        assert str(CustomDataLoader(tmp_path / 'unsupervised.csv')) == f"CustomDataLoader(test_path={tmp_path / 'unsupervised.csv'})"
        assert str(CustomDataLoader(tmp_path / 'semi_supervised.csv', tmp_path / 'semi_supervised_train.csv')) == f"CustomDataLoader(test_path={tmp_path / 'semi_supervised.csv'},train_path={tmp_path / 'semi_supervised_train.csv'})"
        assert str(CustomDataLoader(tmp_path / 'supervised.csv', tmp_path / 'supervised_train.csv')) == f"CustomDataLoader(test_path={tmp_path / 'supervised.csv'},train_path={tmp_path / 'supervised_train.csv'})"

    def test_different_nb_features(self, tmp_path):
        write_supervised_data(tmp_path / 'supervised.csv', tmp_path / 'supervised_train.csv')
        pd.read_csv(tmp_path / 'supervised.csv').drop(columns="f1").to_csv(tmp_path / 'supervised.csv', index=False)
        loader = CustomDataLoader(tmp_path / 'supervised.csv', tmp_path / 'supervised_train.csv')
        with pytest.raises(ValueError):
            loader.load()

    def test_different_feature_names(self, tmp_path):
        write_supervised_data(tmp_path / 'supervised.csv', tmp_path / 'supervised_train.csv')
        pd.read_csv(tmp_path / 'supervised.csv').rename(columns={'f1': 'other-name'}).to_csv(tmp_path / 'supervised.csv', index=False)
        loader = CustomDataLoader(tmp_path / 'supervised.csv', tmp_path / 'supervised_train.csv')
        with pytest.raises(ValueError):
            loader.load()

    def test_shuffled_columns(self, tmp_path):
        write_supervised_data(tmp_path / 'supervised.csv', tmp_path / 'supervised_train.csv')
        pd.read_csv(tmp_path / 'supervised.csv')[['time', 'label', 'f2', 'f3', 'f1']].to_csv(tmp_path / 'supervised.csv', index=False)
        loader = CustomDataLoader(tmp_path / 'supervised.csv', tmp_path / 'supervised_train.csv')
        loader.load()
