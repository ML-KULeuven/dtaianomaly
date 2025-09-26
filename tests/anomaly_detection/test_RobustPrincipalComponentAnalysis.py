import pytest

from dtaianomaly.anomaly_detection import RobustPrincipalComponentAnalysis, Supervision


class TestRobustPrincipalComponentAnalysis:

    def test_supervision(self):
        detector = RobustPrincipalComponentAnalysis(1)
        assert detector.supervision == Supervision.SEMI_SUPERVISED

    def test_initialize_non_int_window_size(self):
        with pytest.raises(TypeError):
            RobustPrincipalComponentAnalysis(window_size=True)
        with pytest.raises(ValueError):
            RobustPrincipalComponentAnalysis(window_size="a string")
        RobustPrincipalComponentAnalysis(5)  # Doesn't raise an error

    def test_initialize_too_small_window_size(self):
        with pytest.raises(ValueError):
            RobustPrincipalComponentAnalysis(window_size=0)
        RobustPrincipalComponentAnalysis(5)  # Doesn't raise an error

    def test_valid_window_size(self):
        RobustPrincipalComponentAnalysis(1)
        RobustPrincipalComponentAnalysis(10)
        RobustPrincipalComponentAnalysis(100)
        RobustPrincipalComponentAnalysis("fft")

    def test_initialize_non_int_max_iter(self):
        with pytest.raises(TypeError):
            RobustPrincipalComponentAnalysis(1, max_iter=True)
        with pytest.raises(TypeError):
            RobustPrincipalComponentAnalysis(1, max_iter="a string")
        with pytest.raises(TypeError):
            RobustPrincipalComponentAnalysis(1, max_iter=0.05)
        RobustPrincipalComponentAnalysis(5)  # Doesn't raise an error

    def test_initialize_too_small_max_iter(self):
        with pytest.raises(ValueError):
            RobustPrincipalComponentAnalysis(1, max_iter=0)
        RobustPrincipalComponentAnalysis(1)  # Doesn't raise an error

    def test_str(self):
        assert (
            str(RobustPrincipalComponentAnalysis(1))
            == "RobustPrincipalComponentAnalysis(window_size=1)"
        )
        assert (
            str(RobustPrincipalComponentAnalysis(1, max_iter=50))
            == "RobustPrincipalComponentAnalysis(window_size=1,max_iter=50)"
        )
