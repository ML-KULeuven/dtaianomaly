import numpy as np

from dtaianomaly.anomaly_detection import ROCKAD, Supervision


class TestROCKAD:

    def test_supervision(self):
        assert ROCKAD(128).supervision == Supervision.UNSUPERVISED

    def test_seed(self, univariate_time_series):
        detector1 = ROCKAD(128, seed=0)
        y_pred1 = detector1.fit(univariate_time_series).decision_function(
            univariate_time_series
        )

        detector2 = ROCKAD(128, seed=0)
        y_pred2 = detector2.fit(univariate_time_series).decision_function(
            univariate_time_series
        )

        assert np.array_equal(y_pred1, y_pred2)
