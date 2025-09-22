from dtaianomaly.anomaly_detection import CopulaBasedOutlierDetector, Supervision


class TestCopulaBasedOutlierDetector:

    def test_supervision(self):
        detector = CopulaBasedOutlierDetector(1)
        assert detector.supervision == Supervision.UNSUPERVISED

    def test_str(self):
        assert (
            str(CopulaBasedOutlierDetector(5))
            == "CopulaBasedOutlierDetector(window_size=5)"
        )
        assert (
            str(CopulaBasedOutlierDetector("fft"))
            == "CopulaBasedOutlierDetector(window_size='fft')"
        )
        assert (
            str(CopulaBasedOutlierDetector(15, 3))
            == "CopulaBasedOutlierDetector(window_size=15,stride=3)"
        )
