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

    @pytest.mark.parametrize("moving_average_window_size", [10, 1, 20, 500])
    def test_moving_average_window_size_valid(self, moving_average_window_size):
        detector = SpectralResidual(moving_average_window_size)
        assert detector.moving_average_window_size == moving_average_window_size

    @pytest.mark.parametrize(
        "moving_average_window_size", [1.0, "0", True, None, ["a", "list"]]
    )
    def test_moving_average_window_size_invalid_type(self, moving_average_window_size):
        with pytest.raises(TypeError):
            SpectralResidual(moving_average_window_size)

    @pytest.mark.parametrize("moving_average_window_size", [0, -1])
    def test_moving_average_window_size_invalid_value(self, moving_average_window_size):
        with pytest.raises(ValueError):
            SpectralResidual(moving_average_window_size)

    @pytest.mark.parametrize("epsilon", [1e-5, 1e-3, 0.1, 1.0, 2.0])
    def test_epsilon_valid(self, epsilon):
        detector = SpectralResidual(3, epsilon)
        assert detector.epsilon == epsilon

    @pytest.mark.parametrize("epsilon", ["0", True, None, ["a", "list"]])
    def test_epsilon_invalid_type(self, epsilon):
        with pytest.raises(TypeError):
            SpectralResidual(3, epsilon)

    @pytest.mark.parametrize("epsilon", [0.0, -1e-16])
    def test_epsilon_invalid_value(self, epsilon):
        with pytest.raises(ValueError):
            SpectralResidual(3, epsilon)

    def test_multivariate(self, univariate_time_series, multivariate_time_series):
        detector = SpectralResidual(16)
        with pytest.raises(ValueError):
            detector.fit(multivariate_time_series)

        detector.fit(univariate_time_series)
        with pytest.raises(ValueError):
            detector.decision_function(multivariate_time_series)
