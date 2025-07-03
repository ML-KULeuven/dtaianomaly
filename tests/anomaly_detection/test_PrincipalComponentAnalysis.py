
import pytest

from dtaianomaly.anomaly_detection import PrincipalComponentAnalysis, Supervision


class TestPrincipalComponentAnalysis:

    def test_supervision(self):
        detector = PrincipalComponentAnalysis(1)
        assert detector.supervision == Supervision.SEMI_SUPERVISED

    def test_str(self):
        assert str(PrincipalComponentAnalysis(5)) == "PrincipalComponentAnalysis(window_size=5)"
        assert str(PrincipalComponentAnalysis('fft')) == "PrincipalComponentAnalysis(window_size='fft')"
        assert str(PrincipalComponentAnalysis(15, 3)) == "PrincipalComponentAnalysis(window_size=15,stride=3)"
        assert str(PrincipalComponentAnalysis(25, n_components=5)) == "PrincipalComponentAnalysis(window_size=25,n_components=5)"

    def test_n_components(self):
        PrincipalComponentAnalysis(15, n_components=1)
        PrincipalComponentAnalysis(15, n_components=2)
        PrincipalComponentAnalysis(15, n_components=4)
        PrincipalComponentAnalysis(15, n_components=8)
        PrincipalComponentAnalysis(15, n_components=0.1)
        PrincipalComponentAnalysis(15, n_components=1.0)
        with pytest.raises(TypeError):
            PrincipalComponentAnalysis(15, n_components='6')
        with pytest.raises(TypeError):
            PrincipalComponentAnalysis(15, n_components=True)
        with pytest.raises(ValueError):
            PrincipalComponentAnalysis(15, n_components=0.0)
        with pytest.raises(ValueError):
            PrincipalComponentAnalysis(15, n_components=-0.1)
        with pytest.raises(ValueError):
            PrincipalComponentAnalysis(15, n_components=1.1)
        with pytest.raises(ValueError):
            PrincipalComponentAnalysis(15, n_components=-1)
        with pytest.raises(ValueError):
            PrincipalComponentAnalysis(15, n_components=0)
