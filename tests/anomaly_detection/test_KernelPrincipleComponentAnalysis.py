
from dtaianomaly.anomaly_detection import KernelPrincipleComponentAnalysis, Supervision


class TestPrincipleComponentAnalysis:

    def test_supervision(self):
        detector = KernelPrincipleComponentAnalysis(1)
        assert detector.supervision == Supervision.SEMI_SUPERVISED

    def test_str(self):
        assert str(KernelPrincipleComponentAnalysis(5)) == "KernelPrincipleComponentAnalysis(window_size=5)"
        assert str(KernelPrincipleComponentAnalysis('fft')) == "KernelPrincipleComponentAnalysis(window_size='fft')"
        assert str(KernelPrincipleComponentAnalysis(15, 3)) == "KernelPrincipleComponentAnalysis(window_size=15,stride=3)"
        assert str(KernelPrincipleComponentAnalysis(25, n_components=5)) == "KernelPrincipleComponentAnalysis(window_size=25,n_components=5)"
        assert str(KernelPrincipleComponentAnalysis(25, kernel='poly')) == "KernelPrincipleComponentAnalysis(window_size=25,kernel='poly')"
