
from dtaianomaly.anomaly_detection import HistogramBasedOutlierScore, Supervision


class TestHistogramBasedOutlierScore:

    def test_supervision(self):
        detector = HistogramBasedOutlierScore(1)
        assert detector.supervision == Supervision.UNSUPERVISED

    def test_str(self):
        assert str(HistogramBasedOutlierScore(5)) == "HistogramBasedOutlierScore(window_size=5)"
        assert str(HistogramBasedOutlierScore('fft')) == "HistogramBasedOutlierScore(window_size='fft')"
        assert str(HistogramBasedOutlierScore(15, 3)) == "HistogramBasedOutlierScore(window_size=15,stride=3)"
        assert str(HistogramBasedOutlierScore(25, n_bins=42)) == "HistogramBasedOutlierScore(window_size=25,n_bins=42)"
        assert str(HistogramBasedOutlierScore(25, alpha=0.5)) == "HistogramBasedOutlierScore(window_size=25,alpha=0.5)"
