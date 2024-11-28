
from dtaianomaly.anomaly_detection import PrincipleComponentAnalysis, Supervision


class TestPrincipleComponentAnalysis:

    def test_supervision(self):
        detector = PrincipleComponentAnalysis(1)
        assert detector.supervision == Supervision.SEMI_SUPERVISED

    def test_str(self):
        assert str(PrincipleComponentAnalysis(5)) == "PrincipleComponentAnalysis(window_size=5)"
        assert str(PrincipleComponentAnalysis('fft')) == "PrincipleComponentAnalysis(window_size='fft')"
        assert str(PrincipleComponentAnalysis(15, 3)) == "PrincipleComponentAnalysis(window_size=15,stride=3)"
        assert str(PrincipleComponentAnalysis(25, n_components=5)) == "PrincipleComponentAnalysis(window_size=25,n_components=5)"
