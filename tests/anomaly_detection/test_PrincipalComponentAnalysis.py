
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
