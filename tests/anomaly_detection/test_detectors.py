
import pytest
import inspect
import numpy as np
from sklearn.exceptions import NotFittedError
from dtaianomaly import anomaly_detection, pipeline, preprocessing, utils


DETECTORS_WITHOUT_FITTING = [
    anomaly_detection.baselines.AlwaysNormal,
    anomaly_detection.baselines.AlwaysAnomalous,
    anomaly_detection.baselines.RandomDetector,
    anomaly_detection.DWT_MLEAD,
    anomaly_detection.MedianMethod
]

DETECTORS_NOT_MULTIVARIATE = [
    anomaly_detection.DWT_MLEAD,
    anomaly_detection.MedianMethod,
    anomaly_detection.KShapeAnomalyDetector
]


def initialize(cls):
    kwargs = {
        'window_size': 15,
        'neighborhood_size_before': 15,
        'detector': anomaly_detection.IsolationForest(window_size=15),
        'preprocessor': preprocessing.Identity(),
    }
    sig = inspect.signature(cls.__init__)
    accepted_params = set(sig.parameters) - {"self"}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_params}
    if cls == anomaly_detection.ClusterBasedLocalOutlierFactor:  # Because sometimes this gives an error based on 'cluster separation'
        filtered_kwargs['random_state'] = 0
    return cls(**filtered_kwargs)


@pytest.mark.parametrize('cls', utils.all_classes('anomaly-detector', return_names=False) + [pipeline.Pipeline])
class TestAnomalyDetectors:

    def test_fit(self, cls, univariate_time_series):
        detector = initialize(cls)
        assert detector.fit(univariate_time_series) is detector

    def test_fit_can_pass_kwargs(self, cls, univariate_time_series):
        initialize(cls).fit(
            univariate_time_series,
            lower_bound=10,
            upper_bound=1000,
            threshold=0.89,
            default_window_size=64
        )

    def test_fit_invalid_array(self, cls):
        with pytest.raises(ValueError):
            initialize(cls).fit([5, 20, '5'])

    def test_decision_function_invalid_array(self, cls, univariate_time_series):
        detector = initialize(cls)
        detector.fit(univariate_time_series)
        with pytest.raises(ValueError):
            detector.decision_function(np.asarray(['foo', 'bar', 'lorem', 'ipsum']))

    def test_decision_function_not_fitted(self, cls, univariate_time_series):
        detector = initialize(cls)
        if type(detector) not in DETECTORS_WITHOUT_FITTING:
            with pytest.raises(NotFittedError):
                detector.decision_function(univariate_time_series)

    def test_decision_function_list_as_input(self, cls, univariate_time_series):
        detector = initialize(cls)
        detector.fit(univariate_time_series)
        decision_function = detector.decision_function([item for item in univariate_time_series])
        assert decision_function.shape[0] == univariate_time_series.shape[0]

    def test_univariate(self, cls, univariate_time_series):
        detector = initialize(cls)
        detector.fit(univariate_time_series)
        decision_function = detector.decision_function(univariate_time_series)
        assert decision_function.shape[0] == univariate_time_series.shape[0]

    def test_univariate_reshape(self, cls, univariate_time_series):
        detector = initialize(cls)
        reshaped_univariate_time_series = univariate_time_series.reshape(-1, 1)
        assert reshaped_univariate_time_series.shape == (univariate_time_series.shape[0], 1)
        detector.fit(reshaped_univariate_time_series)
        decision_function = detector.decision_function(reshaped_univariate_time_series)

    def test_multivariate(self, cls, multivariate_time_series):
        detector = initialize(cls)
        if type(detector) in DETECTORS_NOT_MULTIVARIATE:
            if type(detector) not in DETECTORS_WITHOUT_FITTING:
                with pytest.raises(ValueError):
                    detector.fit(multivariate_time_series)
                detector.fit(multivariate_time_series[:, 0])  # Fit on one dimension only
                with pytest.raises(ValueError):
                    detector.decision_function(multivariate_time_series)
            else:
                with pytest.raises(ValueError):
                    detector.decision_function(multivariate_time_series)
        else:
            detector.fit(multivariate_time_series)
            decision_function = detector.decision_function(multivariate_time_series)
            assert decision_function.shape[0] == multivariate_time_series.shape[0]

    def test_is_valid_array_like_univariate(self, cls, univariate_time_series):
        detector = initialize(cls)
        X_ = detector.fit(univariate_time_series).decision_function(univariate_time_series)
        assert utils.is_valid_array_like(X_)

    def test_is_valid_array_like_multivariate(self, cls, multivariate_time_series):
        detector = initialize(cls)
        if type(detector) not in DETECTORS_NOT_MULTIVARIATE:
            X_ = detector.fit(multivariate_time_series).decision_function(multivariate_time_series)
            assert utils.is_valid_array_like(X_)

    def test_fit_predict_on_different_time_series(self, cls, univariate_time_series):
        # 66% train-test split
        detector = initialize(cls)
        X_train = univariate_time_series[:univariate_time_series.shape[0] * 2 // 3]
        X_test = univariate_time_series[univariate_time_series.shape[0] * 2 // 3:]
        detector.fit(X_train)
        decision_function = detector.predict_proba(X_test)
        assert decision_function.shape[0] == X_test.shape[0]

    def test_predict_confidence(self, cls, univariate_time_series):
        detector = initialize(cls)
        detector.fit(univariate_time_series)
        confidence = detector.predict_confidence(univariate_time_series)
        assert confidence.shape[0] == univariate_time_series.shape[0]
