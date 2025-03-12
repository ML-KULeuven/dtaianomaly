from typing import Optional

import pytest
import numpy as np

import utils
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


class TestFeatureNames:

    def test_no_feature_name_given(self, valid_X_test, valid_y_test):
        data_set = DataSet(X_test=valid_X_test, y_test=valid_y_test)
        assert data_set.feature_names is None

    def test_no_feature_name(self, valid_X_test, valid_y_test):
        feature_names = [f'Feature {i}' for i in range(utils.get_dimension(valid_X_test))]
        data_set = DataSet(X_test=valid_X_test, y_test=valid_y_test, feature_names=feature_names)
        assert data_set.feature_names == feature_names

    def test_invalid_feature_names_given(self, valid_X_test, valid_y_test):
        with pytest.raises(ValueError):
            DataSet(X_test=valid_X_test, y_test=valid_y_test, feature_names=[i for i in range(utils.get_dimension(valid_X_test))])

    def test_invalid_number_of_feature_names_given(self, valid_X_test, valid_y_test):
        with pytest.raises(ValueError):
            DataSet(X_test=valid_X_test, y_test=valid_y_test, feature_names=[f'Feature {i}' for i in range(utils.get_dimension(valid_X_test) + 1)])


class TestTimeSteps:

    def test_no_test_steps(self, valid_X_test, valid_y_test):
        data_set = DataSet(X_test=valid_X_test, y_test=valid_y_test)
        assert data_set.time_steps_test is None

    def test_valid_time_steps_test(self, valid_X_test, valid_y_test):
        time_steps = np.arange(0, valid_X_test.shape[0])
        data_set = DataSet(X_test=valid_X_test, y_test=valid_y_test, time_steps_test=time_steps)
        assert np.array_equal(data_set.time_steps_test, time_steps)

    def test_invalid_time_steps_test(self, valid_X_test, valid_y_test):
        with pytest.raises(ValueError):
            DataSet(X_test=valid_X_test, y_test=valid_y_test, time_steps_test=0)

    def test_time_steps_test_different_size(self, valid_X_test, valid_y_test):
        with pytest.raises(ValueError):
            DataSet(X_test=valid_X_test, y_test=valid_y_test, time_steps_test=np.arange(0, valid_X_test.shape[0] + 1))

    def test_no_train_steps(self, valid_X_test, valid_y_test, valid_X_train):
        data_set = DataSet(X_test=valid_X_test, y_test=valid_y_test, X_train=valid_X_train)
        assert data_set.time_steps_train is None

    def test_valid_train_steps_test(self, valid_X_test, valid_y_test, valid_X_train):
        time_steps = np.arange(0, valid_X_train.shape[0])
        data_set = DataSet(X_test=valid_X_test, y_test=valid_y_test, X_train=valid_X_train, time_steps_train=time_steps)
        assert np.array_equal(data_set.time_steps_train, time_steps)

    def test_invalid_train_steps_test(self, valid_X_test, valid_y_test, valid_X_train):
        with pytest.raises(ValueError):
            DataSet(X_test=valid_X_test, y_test=valid_y_test, X_train=valid_X_train, time_steps_train=0)

    def test_time_steps_train_different_size(self, valid_X_test, valid_y_test, valid_X_train):
        with pytest.raises(ValueError):
            DataSet(X_test=valid_X_test, y_test=valid_y_test, X_train=valid_X_train, time_steps_train=np.arange(0, valid_X_train.shape[0] + 1))

    def test_time_steps_train_no_x_train(self, valid_X_test, valid_y_test):
        with pytest.raises(ValueError):
            DataSet(X_test=valid_X_test, y_test=valid_y_test, time_steps_train=np.arange(0, valid_X_test.shape[0]))

class TestCompatibleSupervision:

    def test_unsupervised_all_data(self, valid_X_test, valid_y_test, valid_X_train, valid_y_train):
        data_set = DataSet(X_test=valid_X_test, y_test=valid_y_test, X_train=valid_X_train, y_train=valid_y_train)
        assert Supervision.UNSUPERVISED in data_set.compatible_supervision()

    def test_unsupervised_no_train_labels(self, valid_X_test, valid_y_test, valid_X_train):
        data_set = DataSet(X_test=valid_X_test, y_test=valid_y_test, X_train=valid_X_train)
        assert Supervision.UNSUPERVISED in data_set.compatible_supervision()

    def test_unsupervised_no_train_data(self, valid_X_test, valid_y_test):
        data_set = DataSet(X_test=valid_X_test, y_test=valid_y_test)
        assert Supervision.UNSUPERVISED in data_set.compatible_supervision()

    def test_semi_supervised_all_data(self, valid_X_test, valid_y_test, valid_X_train, valid_y_train):
        data_set = DataSet(X_test=valid_X_test, y_test=valid_y_test, X_train=valid_X_train, y_train=valid_y_train)
        assert Supervision.SEMI_SUPERVISED in data_set.compatible_supervision()

    def test_semi_supervised_no_train_labels(self, valid_X_test, valid_y_test, valid_X_train):
        data_set = DataSet(X_test=valid_X_test, y_test=valid_y_test, X_train=valid_X_train)
        assert Supervision.SEMI_SUPERVISED in data_set.compatible_supervision()

    def test_semi_supervised_no_train_data(self, valid_X_test, valid_y_test):
        data_set = DataSet(X_test=valid_X_test, y_test=valid_y_test)
        assert Supervision.SEMI_SUPERVISED not in data_set.compatible_supervision()

    def test_supervised_all_data(self, valid_X_test, valid_y_test, valid_X_train, valid_y_train):
        data_set = DataSet(X_test=valid_X_test, y_test=valid_y_test, X_train=valid_X_train, y_train=valid_y_train)
        assert Supervision.SUPERVISED in data_set.compatible_supervision()

    def test_supervised_no_train_labels(self, valid_X_test, valid_y_test, valid_X_train):
        data_set = DataSet(X_test=valid_X_test, y_test=valid_y_test, X_train=valid_X_train)
        assert Supervision.SUPERVISED not in data_set.compatible_supervision()

    def test_supervised_no_train_data(self, valid_X_test, valid_y_test):
        data_set = DataSet(X_test=valid_X_test, y_test=valid_y_test)
        assert Supervision.SUPERVISED not in data_set.compatible_supervision()


class DummyDetector(BaseDetector):

    def __init__(self, supervision):
        super().__init__(supervision)

    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> None:
        pass

    def _decision_function(self, X: np.ndarray) -> np.array:
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
