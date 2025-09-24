from dtaianomaly.evaluation import Precision, ThresholdMetric
from dtaianomaly.thresholding import FixedCutoffThreshold


class TestThresholding:

    def test_piped_str(self):
        assert (
            ThresholdMetric(FixedCutoffThreshold(0.5), Precision()).piped_str()
            == "FixedCutoffThreshold(cutoff=0.5)->Precision()"
        )
