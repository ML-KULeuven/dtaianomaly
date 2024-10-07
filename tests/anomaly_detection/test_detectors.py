
import pytest
import copy
import numpy as np
from sklearn.exceptions import NotFittedError
from dtaianomaly import anomaly_detection, pipeline, preprocessing, utils


DETECTORS_WITHOUT_FITTING = [anomaly_detection.MatrixProfileDetector]


@pytest.fixture(params=[
    anomaly_detection.IsolationForest(15),
    anomaly_detection.LocalOutlierFactor(15),
    anomaly_detection.MatrixProfileDetector(15),
    pipeline.Pipeline(preprocessing.Identity(), anomaly_detection.IsolationForest(15))
])
def detector(request):
    return copy.deepcopy(request.param)


class TestAnomalyDetectors:

    def test_fit(self, detector, univariate_time_series):
        assert detector.fit(univariate_time_series) is detector

    def test_fit_invalid_array(self, detector):
        if type(detector) in DETECTORS_WITHOUT_FITTING:
            # If no fitting is needed, then no check should be performed
            detector.fit([5, 20, '5'])
        else:
            with pytest.raises(ValueError):
                detector.fit([5, 20, '5'])

    def test_decision_function_invalid_array(self, detector, univariate_time_series):
        detector.fit(univariate_time_series)
        with pytest.raises(ValueError):
            detector.decision_function(np.asarray(['foo', 'bar', 'lorem', 'ipsum']))

    def test_decision_function_not_fitted(self, detector, univariate_time_series):
        if type(detector) not in DETECTORS_WITHOUT_FITTING:
            with pytest.raises(NotFittedError):
                detector.decision_function(univariate_time_series)

    def test_decision_function_list_as_input(self, detector, univariate_time_series):
        detector.fit(univariate_time_series)
        decision_function = detector.decision_function([item for item in univariate_time_series])
        assert decision_function.shape[0] == univariate_time_series.shape[0]

    def test_univariate(self, detector, univariate_time_series):
        detector.fit(univariate_time_series)
        decision_function = detector.decision_function(univariate_time_series)
        assert decision_function.shape[0] == univariate_time_series.shape[0]

    def test_univariate_reshape(self, detector, univariate_time_series):
        reshaped_univariate_time_series = univariate_time_series.reshape(-1, 1)
        assert reshaped_univariate_time_series.shape == (univariate_time_series.shape[0], 1)
        detector.fit(reshaped_univariate_time_series)
        decision_function = detector.decision_function(reshaped_univariate_time_series)

    def test_multivariate(self, detector, multivariate_time_series):
        detector.fit(multivariate_time_series)
        decision_function = detector.decision_function(multivariate_time_series)
        assert decision_function.shape[0] == multivariate_time_series.shape[0]

    def test_is_valid_array_like_univariate(self, detector, univariate_time_series):
        X_ = detector.fit(univariate_time_series).decision_function(univariate_time_series)
        assert utils.is_valid_array_like(X_)

    def test_is_valid_array_like_multivariate(self, detector, multivariate_time_series):
        X_ = detector.fit(multivariate_time_series).decision_function(multivariate_time_series)
        assert utils.is_valid_array_like(X_)

    def test_fit_predict_on_different_time_series(self, detector, univariate_time_series):
        # 66% train-test split
        X_train = univariate_time_series[:univariate_time_series.shape[0] * 2 // 3]
        X_test = univariate_time_series[univariate_time_series.shape[0] * 2 // 3:]
        detector.fit(X_train)
        decision_function = detector.predict_proba(X_test)
        assert decision_function.shape[0] == X_test.shape[0]