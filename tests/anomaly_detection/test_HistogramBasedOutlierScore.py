import pytest

from dtaianomaly.anomaly_detection import HistogramBasedOutlierScore, Supervision


class TestHistogramBasedOutlierScore:

    def test_supervision(self):
        detector = HistogramBasedOutlierScore(1)
        assert detector.supervision == Supervision.UNSUPERVISED

    def test_str(self):
        assert (
            str(HistogramBasedOutlierScore(5))
            == "HistogramBasedOutlierScore(window_size=5)"
        )
        assert (
            str(HistogramBasedOutlierScore("fft"))
            == "HistogramBasedOutlierScore(window_size='fft')"
        )
        assert (
            str(HistogramBasedOutlierScore(15, 3))
            == "HistogramBasedOutlierScore(window_size=15,stride=3)"
        )
        assert (
            str(HistogramBasedOutlierScore(25, n_bins=42))
            == "HistogramBasedOutlierScore(window_size=25,n_bins=42)"
        )
        assert (
            str(HistogramBasedOutlierScore(25, alpha=0.5))
            == "HistogramBasedOutlierScore(window_size=25,alpha=0.5)"
        )

    def test_n_bins(self):
        HistogramBasedOutlierScore(15, n_bins="auto")
        HistogramBasedOutlierScore(15, n_bins=2)
        HistogramBasedOutlierScore(15, n_bins=4)
        HistogramBasedOutlierScore(15, n_bins=8)
        with pytest.raises(TypeError):
            HistogramBasedOutlierScore(15, n_bins=5.0)
        with pytest.raises(TypeError):
            HistogramBasedOutlierScore(15, n_bins=True)
        with pytest.raises(ValueError):
            HistogramBasedOutlierScore(15, n_bins=1)
        with pytest.raises(ValueError):
            HistogramBasedOutlierScore(15, n_bins=0)
        with pytest.raises(ValueError):
            HistogramBasedOutlierScore(15, n_bins="some-invalid-string")

    def test_alpha(self):
        HistogramBasedOutlierScore(15, alpha=0.01)
        HistogramBasedOutlierScore(15, alpha=0.5)
        HistogramBasedOutlierScore(15, alpha=0.99)
        with pytest.raises(TypeError):
            HistogramBasedOutlierScore(15, alpha="6")
        with pytest.raises(TypeError):
            HistogramBasedOutlierScore(15, alpha=True)
        with pytest.raises(ValueError):
            HistogramBasedOutlierScore(15, alpha=-0.1)
        with pytest.raises(ValueError):
            HistogramBasedOutlierScore(15, alpha=0.0)
        with pytest.raises(ValueError):
            HistogramBasedOutlierScore(15, alpha=1.0)
        with pytest.raises(ValueError):
            HistogramBasedOutlierScore(15, alpha=1.1)

    def test_tol(self):
        HistogramBasedOutlierScore(15, tol=0.01)
        HistogramBasedOutlierScore(15, tol=0.5)
        HistogramBasedOutlierScore(15, tol=0.99)
        with pytest.raises(TypeError):
            HistogramBasedOutlierScore(15, tol="6")
        with pytest.raises(TypeError):
            HistogramBasedOutlierScore(15, tol=True)
        with pytest.raises(ValueError):
            HistogramBasedOutlierScore(15, tol=-0.1)
        with pytest.raises(ValueError):
            HistogramBasedOutlierScore(15, tol=0.0)
        with pytest.raises(ValueError):
            HistogramBasedOutlierScore(15, tol=1.0)
        with pytest.raises(ValueError):
            HistogramBasedOutlierScore(15, tol=1.1)
