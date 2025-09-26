import warnings
from typing import Optional

import numpy as np
import pytest

from dtaianomaly.anomaly_detection import (
    BaseDetector,
    MultivariateDetector,
    RandomDetector,
    Supervision,
)


class DummyDetector(BaseDetector):

    def __init__(self, supervision: Supervision = Supervision.UNSUPERVISED):
        super().__init__(supervision)

    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> None:
        raise NotImplementedError

    def _decision_function(self, X: np.ndarray) -> np.array:
        raise NotImplementedError


class TestMultivariateDetector:

    @pytest.mark.parametrize(
        "detector",
        [
            DummyDetector(Supervision.UNSUPERVISED),
            DummyDetector(Supervision.SEMI_SUPERVISED),
            DummyDetector(Supervision.SUPERVISED),
        ],
    )
    def test_supervision(self, detector):
        multivariate_detector = MultivariateDetector(detector)
        assert multivariate_detector.supervision == detector.supervision

    def test_initialize_invalid_detector(self):
        with pytest.raises(TypeError):
            MultivariateDetector("DummyDetector()")
        MultivariateDetector(DummyDetector())  # Doesn't raise an error

    def test_initialize_invalid_type_aggregation(self):
        with pytest.raises(TypeError):
            MultivariateDetector(DummyDetector(), 0)
        with pytest.raises(TypeError):
            MultivariateDetector(DummyDetector(), True)
        with pytest.raises(TypeError):
            MultivariateDetector(DummyDetector(), 0.5)

    def test_initialize_invalid_aggregation(self):
        with pytest.raises(ValueError):
            MultivariateDetector(DummyDetector(), "something-invalid")
        MultivariateDetector(DummyDetector(), "mean")
        MultivariateDetector(DummyDetector(), "max")
        MultivariateDetector(DummyDetector(), "min")

    @pytest.mark.parametrize("aggregation", ["mean", "min", "max"])
    def test_aggregations(self, aggregation, multivariate_time_series):
        detector = MultivariateDetector(RandomDetector(seed=0), aggregation)
        detector.fit(multivariate_time_series)
        detector.decision_function(multivariate_time_series)

    def test_initialize_non_bool_raise_warning_for_univariate(self):
        with pytest.raises(TypeError):
            MultivariateDetector(DummyDetector(), raise_warning_for_univariate=0)
        with pytest.raises(TypeError):
            MultivariateDetector(DummyDetector(), raise_warning_for_univariate="True")
        MultivariateDetector(DummyDetector(), raise_warning_for_univariate=True)

    def test_warning_univariate_raise(self, univariate_time_series):
        detector = MultivariateDetector(
            RandomDetector(), raise_warning_for_univariate=True
        )
        with pytest.warns(UserWarning):
            detector.fit(univariate_time_series)

    def test_warning_univariate_no_raise(self, univariate_time_series):
        detector = MultivariateDetector(
            RandomDetector(), raise_warning_for_univariate=False
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            detector.fit(univariate_time_series)

    def test_warning_multivariate_raise(self, multivariate_time_series):
        detector = MultivariateDetector(
            RandomDetector(), raise_warning_for_univariate=True
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            detector.fit(multivariate_time_series)

    def test_warning_multivariate_no_raise(self, multivariate_time_series):
        detector = MultivariateDetector(
            RandomDetector(), raise_warning_for_univariate=False
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            detector.fit(multivariate_time_series)

    def test_different_fit_predict_dimension(
        self, multivariate_time_series, univariate_time_series
    ):
        detector = MultivariateDetector(RandomDetector())
        detector.fit(univariate_time_series)
        with pytest.raises(ValueError):
            detector.decision_function(multivariate_time_series)

    def test_str(self):
        assert (
            str(MultivariateDetector(DummyDetector()))
            == "MultivariateDetector(detector=DummyDetector())"
        )
        assert (
            str(MultivariateDetector(DummyDetector(), "min"))
            == "MultivariateDetector(detector=DummyDetector(),aggregation='min')"
        )
