import pytest

from dtaianomaly.anomaly_detection import KernelPrincipalComponentAnalysis, Supervision


class TestKernelPrincipalComponentAnalysis:

    def test_supervision(self):
        detector = KernelPrincipalComponentAnalysis(1)
        assert detector.supervision == Supervision.SEMI_SUPERVISED

    def test_str(self):
        assert (
            str(KernelPrincipalComponentAnalysis(5))
            == "KernelPrincipalComponentAnalysis(window_size=5)"
        )
        assert (
            str(KernelPrincipalComponentAnalysis("fft"))
            == "KernelPrincipalComponentAnalysis(window_size='fft')"
        )
        assert (
            str(KernelPrincipalComponentAnalysis(15, 3))
            == "KernelPrincipalComponentAnalysis(window_size=15,stride=3)"
        )
        assert (
            str(KernelPrincipalComponentAnalysis(25, n_components=5))
            == "KernelPrincipalComponentAnalysis(window_size=25,n_components=5)"
        )
        assert (
            str(KernelPrincipalComponentAnalysis(25, kernel="poly"))
            == "KernelPrincipalComponentAnalysis(window_size=25,kernel='poly')"
        )

    def test_n_components(self):
        KernelPrincipalComponentAnalysis(15, n_components=1)
        KernelPrincipalComponentAnalysis(15, n_components=2)
        KernelPrincipalComponentAnalysis(15, n_components=4)
        KernelPrincipalComponentAnalysis(15, n_components=8)
        with pytest.raises(TypeError):
            KernelPrincipalComponentAnalysis(15, n_components="6")
        with pytest.raises(TypeError):
            KernelPrincipalComponentAnalysis(15, n_components=True)
        with pytest.raises(TypeError):
            KernelPrincipalComponentAnalysis(15, n_components=0.0)
        with pytest.raises(ValueError):
            KernelPrincipalComponentAnalysis(15, n_components=-1)
        with pytest.raises(ValueError):
            KernelPrincipalComponentAnalysis(15, n_components=0)

    def test_kernel(self):
        KernelPrincipalComponentAnalysis(15, kernel="sigmoid")
        KernelPrincipalComponentAnalysis(15, kernel="linear")
        KernelPrincipalComponentAnalysis(15, kernel="rbf")
        KernelPrincipalComponentAnalysis(15, kernel="poly")
        KernelPrincipalComponentAnalysis(15, kernel="cosine")
        with pytest.raises(TypeError):
            KernelPrincipalComponentAnalysis(15, kernel=0.5)
        with pytest.raises(TypeError):
            KernelPrincipalComponentAnalysis(15, kernel=True)
        with pytest.raises(TypeError):
            KernelPrincipalComponentAnalysis(15, kernel=1)
        with pytest.raises(ValueError):
            KernelPrincipalComponentAnalysis(15, kernel="something-invalid")
