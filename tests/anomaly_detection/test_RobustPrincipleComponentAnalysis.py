
import pytest
from dtaianomaly.anomaly_detection import RobustPrincipleComponentAnalysis, Supervision


class TestPrincipleComponentAnalysis:

    def test_supervision(self):
        detector = RobustPrincipleComponentAnalysis()
        assert detector.supervision == Supervision.SEMI_SUPERVISED

    def test_initialize_non_int_max_iter(self):
        with pytest.raises(TypeError):
            RobustPrincipleComponentAnalysis(max_iter=True)
        with pytest.raises(TypeError):
            RobustPrincipleComponentAnalysis(max_iter='a string')
        with pytest.raises(TypeError):
            RobustPrincipleComponentAnalysis(max_iter=0.05)
        RobustPrincipleComponentAnalysis(5)  # Doesn't raise an error

    def test_initialize_too_small_max_iter(self):
        with pytest.raises(ValueError):
            RobustPrincipleComponentAnalysis(max_iter=0)
        RobustPrincipleComponentAnalysis(1)  # Doesn't raise an error

    def test_initialize_non_bool_zero_pruning(self):
        with pytest.raises(TypeError):
            RobustPrincipleComponentAnalysis(zero_pruning=1)
        with pytest.raises(TypeError):
            RobustPrincipleComponentAnalysis(zero_pruning=0.0)
        with pytest.raises(TypeError):
            RobustPrincipleComponentAnalysis(zero_pruning='True')
        RobustPrincipleComponentAnalysis(zero_pruning=True)  # Doesn't raise an error

    def test_str(self):
        assert str(RobustPrincipleComponentAnalysis()) == "RobustPrincipleComponentAnalysis()"
        assert str(RobustPrincipleComponentAnalysis(50)) == "RobustPrincipleComponentAnalysis(max_iter=50)"
        assert str(RobustPrincipleComponentAnalysis(50, False)) == "RobustPrincipleComponentAnalysis(max_iter=50,zero_pruning=False)"
        assert str(RobustPrincipleComponentAnalysis(zero_pruning=False)) == "RobustPrincipleComponentAnalysis(zero_pruning=False)"
