
from dtaianomaly.anomaly_detection import HistogramBasedOutlierScore


class TestHistogramBasedOutlierScore:

    def test_str(self):
        assert str(HistogramBasedOutlierScore(5)) == "HistogramBasedOutlierScore(window_size=5)"
        assert str(HistogramBasedOutlierScore('fft')) == "HistogramBasedOutlierScore(window_size='fft')"
        assert str(HistogramBasedOutlierScore(15, 3)) == "HistogramBasedOutlierScore(window_size=15,stride=3)"
        assert str(HistogramBasedOutlierScore(25, n_bins=42)) == "HistogramBasedOutlierScore(window_size=25,n_bins=42)"
        assert str(HistogramBasedOutlierScore(25, alpha=0.5)) == "HistogramBasedOutlierScore(window_size=25,alpha=0.5)"
