import pytest

from dtaianomaly.anomaly_detection import SpectralResidual, Supervision


class TestMedianMethod:

    def test_supervision(self):
        detector = SpectralResidual(15)
        assert detector.supervision == Supervision.UNSUPERVISED

    def test_str(self):
        assert (
            str(SpectralResidual(5)) == "SpectralResidual(moving_average_window_size=5)"
        )
        assert (
            str(SpectralResidual(5, epsilon=0.1))
            == "SpectralResidual(moving_average_window_size=5,epsilon=0.1)"
        )

    def test_moving_average_window_size_valid(self):
        detector = SpectralResidual(1)
        assert detector.moving_average_window_size == 1

    def test_moving_average_window_size_invalid_type(self):
        with pytest.raises(TypeError):
            SpectralResidual(1.0)

    def test_moving_average_window_size_invalid_value(self):
        with pytest.raises(ValueError):
            SpectralResidual(0)

    def test_epsilon_valid(self):
        detector = SpectralResidual(3, 1e-5)
        assert detector.epsilon == 1e-5

    def test_epsilon_invalid_type(self):
        with pytest.raises(TypeError):
            SpectralResidual(3, "0")

    def test_epsilon_invalid_value(self):
        with pytest.raises(ValueError):
            SpectralResidual(3, 0.0)

    def test_multivariate(self, univariate_time_series, multivariate_time_series):
        detector = SpectralResidual(16)
        with pytest.raises(ValueError):
            detector.fit(multivariate_time_series)

        detector.fit(univariate_time_series)
        with pytest.raises(ValueError):
            detector.decision_function(multivariate_time_series)
