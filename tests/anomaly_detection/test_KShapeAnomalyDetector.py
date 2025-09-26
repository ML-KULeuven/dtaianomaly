import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from dtaianomaly.anomaly_detection import KShapeAnomalyDetector, Supervision


class TestKShapeAnomalyDetector:

    def test_supervision(self):
        detector = KShapeAnomalyDetector(1)
        assert detector.supervision == Supervision.UNSUPERVISED

    def test_initialize_non_int_window_size(self):
        with pytest.raises(TypeError):
            KShapeAnomalyDetector(window_size=True)
        with pytest.raises(ValueError):
            KShapeAnomalyDetector(window_size="a string")
        KShapeAnomalyDetector(5)  # Doesn't raise an error

    def test_initialize_too_small_window_size(self):
        with pytest.raises(ValueError):
            KShapeAnomalyDetector(window_size=0)
        KShapeAnomalyDetector(5)  # Doesn't raise an error

    def test_valid_window_size(self):
        KShapeAnomalyDetector(1)
        KShapeAnomalyDetector(10)
        KShapeAnomalyDetector(100)
        KShapeAnomalyDetector("fft")

    def test_n_clusters(self):
        KShapeAnomalyDetector(15, n_clusters=2)
        KShapeAnomalyDetector(15, n_clusters=4)
        KShapeAnomalyDetector(15, n_clusters=8)
        with pytest.raises(TypeError):
            KShapeAnomalyDetector(15, n_clusters="6")
        with pytest.raises(TypeError):
            KShapeAnomalyDetector(15, n_clusters=6.0)
        with pytest.raises(TypeError):
            KShapeAnomalyDetector(15, n_clusters=True)
        with pytest.raises(ValueError):
            KShapeAnomalyDetector(15, n_clusters=0)
        with pytest.raises(ValueError):
            KShapeAnomalyDetector(15, n_clusters=-1)
        with pytest.raises(ValueError):
            KShapeAnomalyDetector(15, n_clusters=1)

    def test_initialize_non_float_sequence_length_multiplier(self):
        with pytest.raises(TypeError):
            KShapeAnomalyDetector(window_size=15, sequence_length_multiplier=True)
        with pytest.raises(TypeError):
            KShapeAnomalyDetector(window_size=15, sequence_length_multiplier="A string")
        KShapeAnomalyDetector(
            5, sequence_length_multiplier=2.5
        )  # Doesn't raise an error
        KShapeAnomalyDetector(
            5, sequence_length_multiplier=3.0
        )  # Doesn't raise an error

    def test_initialize_too_small_sequance_length_multiplier(self):
        with pytest.raises(ValueError):
            KShapeAnomalyDetector(window_size=15, sequence_length_multiplier=0.5)
        with pytest.raises(ValueError):
            KShapeAnomalyDetector(window_size=15, sequence_length_multiplier=0.0)
        with pytest.raises(ValueError):
            KShapeAnomalyDetector(window_size=15, sequence_length_multiplier=0.9999)
        KShapeAnomalyDetector(
            window_size=15, sequence_length_multiplier=1.0
        )  # Doesn't raise an error with float

    def test_initialize_non_float_overlap_rate(self):
        with pytest.raises(TypeError):
            KShapeAnomalyDetector(window_size=15, overlap_rate=True)
        with pytest.raises(TypeError):
            KShapeAnomalyDetector(window_size=15, overlap_rate="A string")
        with pytest.raises(TypeError):
            KShapeAnomalyDetector(window_size=15, overlap_rate=1)
        KShapeAnomalyDetector(5, overlap_rate=0.5)  # Doesn't raise an error

    def test_initialize_invalid_overlap_rate(self):
        with pytest.raises(ValueError):
            KShapeAnomalyDetector(window_size=15, overlap_rate=1.00001)
        with pytest.raises(ValueError):
            KShapeAnomalyDetector(window_size=15, overlap_rate=0.0)
        KShapeAnomalyDetector(5, overlap_rate=0.5)  # Doesn't raise an error

    def test_invalid_additional_arguments(self):
        with pytest.raises(TypeError):
            KShapeAnomalyDetector(window_size="fft", some_invalid_arg=1)
        KShapeAnomalyDetector(window_size="fft", n_clusters=10)

    def test_theta_not_fitted(self, univariate_time_series):
        detector = KShapeAnomalyDetector(window_size=15)
        with pytest.raises(NotFittedError):
            detector.theta_()

    @pytest.mark.slow
    def test_theta(self, univariate_time_series):
        detector = KShapeAnomalyDetector(window_size=15)
        detector.fit(univariate_time_series)
        theta = detector.theta_()
        for i in range(len(theta)):
            assert np.array_equal(detector.centroids_[i], theta[i][0])
            assert detector.weights_[i] == theta[i][1]
