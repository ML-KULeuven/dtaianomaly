from typing import Optional

import pytest
import numpy as np
from dtaianomaly.data import DataSet
from dtaianomaly.anomaly_detection import BaseDetector,  Supervision


@pytest.fixture
def valid_X_test():
    return np.array([0, 1, 2, 3, 4, 5])


@pytest.fixture
def valid_y_test():
    return np.array([0, 0, 1, 0, 1, 0])


@pytest.fixture
def valid_X_train():
    return np.array([0, 10, 20, 30, 40, 50, 60])


@pytest.fixture
def valid_y_train():
    return np.array([0, 1, 1, 0, 0, 0, 0])


class TestCheckIsValid:

    def test_is_valid(self, valid_X_test, valid_y_test):
        data_set = DataSet(X_test=valid_X_test, y_test=valid_y_test)
        assert data_set.is_valid()

    def test_is_valid_given_train_data(self, valid_X_test, valid_y_test, valid_X_train):
        data_set = DataSet(X_test=valid_X_test, y_test=valid_y_test, X_train=valid_X_train)
        assert data_set.is_valid()

    def test_is_valid_given_train_data_and_labels(self, valid_X_test, valid_y_test, valid_X_train, valid_y_train):
        data_set = DataSet(X_test=valid_X_test, y_test=valid_y_test, X_train=valid_X_train, y_train=valid_y_train)
        assert data_set.is_valid()

    def test_not_is_valid(self, valid_X_test, valid_y_test):
        data_set = DataSet(X_test=valid_X_test, y_test=valid_y_test)
        assert data_set.is_valid()
        data_set.X_test = [0, 1, 2, 3, '4', 5]  # Make data invalid (should not be done!)
        assert not data_set.is_valid()

    def test_no_test_data(self, valid_y_test):
        with pytest.raises(ValueError):
            DataSet(X_test=None, y_test=valid_y_test)

    def test_no_test_labels(self, valid_X_test):
        with pytest.raises(ValueError):
            DataSet(X_test=valid_X_test, y_test=None)

    def test_invalid_X_test(self, valid_y_test):
        with pytest.raises(ValueError):
            DataSet(X_test=np.array([0, 1, 2, 3, '4', 5]), y_test=valid_y_test)

    def test_invalid_y_test(self, valid_X_test):
        with pytest.raises(ValueError):
            DataSet(X_test=valid_X_test, y_test=np.array([0, 0, 1, 0, '1', 0]))

    def test_multivariate_y_test(self, valid_X_test):
        with pytest.raises(ValueError):
            DataSet(X_test=valid_X_test, y_test=np.array([[0, 0], [0, 1], [1, 1], [0, 0], [1, 1], [0, 0]]))

    def test_non_binary_y_test(self, valid_X_test):
        with pytest.raises(ValueError):
            DataSet(X_test=valid_X_test, y_test=np.array([0, 0, 1, 0.5, 1, 0]))

    def test_test_data_and_labels_different_shape(self, valid_X_test, valid_y_test):
        with pytest.raises(ValueError):
            DataSet(X_test=valid_X_test, y_test=np.append(valid_y_test, 0))

    def test_invalid_X_train(self, valid_X_test, valid_y_test):
        with pytest.raises(ValueError):
            DataSet(X_test=valid_X_test, y_test=valid_y_test, X_train=np.array([0, 1, 2, 3, '4', 5]))

    def test_X_test_and_X_train_different_dimension(self, valid_X_test, valid_y_test, valid_X_train):
        multivariate_valid_X_train = np.repeat(valid_X_train, 2).reshape(valid_X_train.shape[0], -1)
        with pytest.raises(ValueError):
            DataSet(X_test=valid_X_test, y_test=valid_y_test, X_train=multivariate_valid_X_train)

    def test_valid_y_train_but_no_X_train(self, valid_X_test, valid_y_test, valid_y_train):
        with pytest.raises(ValueError):
            DataSet(X_test=valid_X_test, y_test=valid_y_test, y_train=valid_y_train)

    def test_invalid_y_train(self, valid_X_test, valid_y_test, valid_X_train):
        with pytest.raises(ValueError):
            DataSet(X_test=valid_X_test, y_test=valid_y_test, X_train=valid_X_train, y_train=np.array([0, 0, 1, 0, '1', 0, 0]))

    def test_multivariate_y_train(self, valid_X_test, valid_y_test, valid_X_train):
        with pytest.raises(ValueError):
            DataSet(X_test=valid_X_test, y_test=valid_y_test, X_train=valid_X_train, y_train=np.array([[0, 0], [0, 1], [1, 1], [0, 0], [1, 1], [0, 0], [1, 0]]))

    def test_non_binary_y_train(self, valid_X_test, valid_y_test, valid_X_train):
        with pytest.raises(ValueError):
            DataSet(X_test=valid_X_test, y_test=valid_y_test, X_train=valid_X_train, y_train=np.array([0, 0, 1, 0.5, 1, 0, 0]))

    def test_train_data_and_labels_different_shape(self, valid_X_test, valid_y_test, valid_X_train, valid_y_train):
        with pytest.raises(ValueError):
            DataSet(X_test=valid_X_test, y_test=valid_y_test, X_train=valid_X_train, y_train=np.append(valid_y_train, 0))


class DummyDetector(BaseDetector):

    def __init__(self, supervision):
        super().__init__(supervision)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'BaseDetector':
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0])


@pytest.fixture
def unsupervised_detector() -> DummyDetector:
    return DummyDetector(Supervision.UNSUPERVISED)


@pytest.fixture
def semisupervised_detector() -> DummyDetector:
    return DummyDetector(Supervision.SEMI_SUPERVISED)


@pytest.fixture
def supervised_detector() -> DummyDetector:
    return DummyDetector(Supervision.SUPERVISED)


class TestCompatibleAnomalyDetector:

    def test_no_train_data(self,
                           unsupervised_detector,
                           semisupervised_detector,
                           supervised_detector,
                           valid_X_test,
                           valid_y_test):
        data_set = DataSet(X_test=valid_X_test, y_test=valid_y_test)
        assert data_set.is_compatible(unsupervised_detector)
        assert not data_set.is_compatible(semisupervised_detector)
        assert not data_set.is_compatible(supervised_detector)

    def test_no_train_labels(self,
                             unsupervised_detector,
                             semisupervised_detector,
                             supervised_detector,
                             valid_X_test,
                             valid_y_test,
                             valid_X_train):
        data_set = DataSet(X_test=valid_X_test, y_test=valid_y_test, X_train=valid_X_train)
        assert data_set.is_compatible(unsupervised_detector)
        assert data_set.is_compatible(semisupervised_detector)
        assert not data_set.is_compatible(supervised_detector)

    def test_train_data(self,
                        unsupervised_detector,
                        semisupervised_detector,
                        supervised_detector,
                        valid_X_test,
                        valid_y_test,
                        valid_X_train,
                        valid_y_train):
        data_set = DataSet(X_test=valid_X_test, y_test=valid_y_test, X_train=valid_X_train, y_train=valid_y_train)
        assert data_set.is_compatible(unsupervised_detector)
        assert data_set.is_compatible(semisupervised_detector)
        assert data_set.is_compatible(supervised_detector)
