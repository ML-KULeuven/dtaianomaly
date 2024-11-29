
from dtaianomaly.anomaly_detection import KernelPrincipalComponentAnalysis, Supervision


class TestKernelPrincipalComponentAnalysis:

    def test_supervision(self):
        detector = KernelPrincipalComponentAnalysis(1)
        assert detector.supervision == Supervision.SEMI_SUPERVISED

    def test_str(self):
        assert str(KernelPrincipalComponentAnalysis(5)) == "KernelPrincipalComponentAnalysis(window_size=5)"
        assert str(KernelPrincipalComponentAnalysis('fft')) == "KernelPrincipalComponentAnalysis(window_size='fft')"
        assert str(KernelPrincipalComponentAnalysis(15, 3)) == "KernelPrincipalComponentAnalysis(window_size=15,stride=3)"
        assert str(KernelPrincipalComponentAnalysis(25, n_components=5)) == "KernelPrincipalComponentAnalysis(window_size=25,n_components=5)"
        assert str(KernelPrincipalComponentAnalysis(25, kernel='poly')) == "KernelPrincipalComponentAnalysis(window_size=25,kernel='poly')"
