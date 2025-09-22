import numpy as np

from dtaianomaly.anomaly_detection import Supervision, baselines


class TestAlwaysNormal:

    def test_supervision(self):
        detector = baselines.AlwaysNormal()
        assert detector.supervision == Supervision.UNSUPERVISED

    def test_univariate(self, univariate_time_series):
        detector = baselines.AlwaysNormal()
        y_pred = detector.fit(univariate_time_series).decision_function(
            univariate_time_series
        )
        assert np.all(y_pred == 0.0)
        assert y_pred.shape == (univariate_time_series.shape[0],)

    def test_multivariate(self, multivariate_time_series):
        detector = baselines.AlwaysNormal()
        y_pred = detector.fit(multivariate_time_series).decision_function(
            multivariate_time_series
        )
        assert np.all(y_pred == 0.0)
        assert y_pred.shape == (multivariate_time_series.shape[0],)

    def test_predict_proba(self, univariate_time_series):
        detector = baselines.AlwaysNormal()
        y_pred = detector.fit(univariate_time_series).predict_proba(
            univariate_time_series
        )
        assert np.all(y_pred == 0.0)
        assert y_pred.shape == (univariate_time_series.shape[0],)

    def test_str(self):
        assert str(baselines.AlwaysNormal()) == "AlwaysNormal()"


class TestAlwaysAnomalous:

    def test_supervision(self):
        detector = baselines.AlwaysAnomalous()
        assert detector.supervision == Supervision.UNSUPERVISED

    def test_univariate(self, univariate_time_series):
        detector = baselines.AlwaysAnomalous()
        y_pred = detector.fit(univariate_time_series).decision_function(
            univariate_time_series
        )
        assert np.all(y_pred == 1.0)
        assert y_pred.shape == (univariate_time_series.shape[0],)

    def test_multivariate(self, multivariate_time_series):
        detector = baselines.AlwaysAnomalous()
        y_pred = detector.fit(multivariate_time_series).decision_function(
            multivariate_time_series
        )
        assert np.all(y_pred == 1.0)
        assert y_pred.shape == (multivariate_time_series.shape[0],)

    def test_predict_proba(self, univariate_time_series):
        detector = baselines.AlwaysAnomalous()
        y_pred = detector.fit(univariate_time_series).predict_proba(
            univariate_time_series
        )
        assert np.all(y_pred == 1.0)
        assert y_pred.shape == (univariate_time_series.shape[0],)

    def test_str(self):
        assert str(baselines.AlwaysAnomalous()) == "AlwaysAnomalous()"


class TestRandomDetector:

    def test_supervision(self):
        detector = baselines.RandomDetector()
        assert detector.supervision == Supervision.UNSUPERVISED

    def test_univariate(self, univariate_time_series):
        detector = baselines.RandomDetector()
        y_pred = detector.fit(univariate_time_series).decision_function(
            univariate_time_series
        )
        assert np.all((0.0 <= y_pred) & (y_pred <= 1.0))
        assert y_pred.shape == (univariate_time_series.shape[0],)

    def test_multivariate(self, multivariate_time_series):
        detector = baselines.AlwaysAnomalous()
        y_pred = detector.fit(multivariate_time_series).decision_function(
            multivariate_time_series
        )
        assert np.all((0.0 <= y_pred) & (y_pred <= 1.0))
        assert y_pred.shape == (multivariate_time_series.shape[0],)

    def test_predict_proba(self, univariate_time_series):
        detector = baselines.AlwaysAnomalous()
        y_pred = detector.fit(univariate_time_series).predict_proba(
            univariate_time_series
        )
        assert np.all((0.0 <= y_pred) & (y_pred <= 1.0))
        assert y_pred.shape == (univariate_time_series.shape[0],)

    def test_seed(self, univariate_time_series):
        detector = baselines.RandomDetector(seed=42)
        random_numbers = []
        for _ in range(10):
            detector.fit(univariate_time_series).decision_function(
                univariate_time_series
            )
            random_numbers.append(np.random.uniform())
        # Not all numbers are the same, so different random numbers should be generated
        assert len(set(random_numbers)) > 1

    def test_no_seed(self, univariate_time_series):
        detector = baselines.RandomDetector()
        y_pred_1 = detector.fit(univariate_time_series).decision_function(
            univariate_time_series
        )
        y_pred_2 = detector.fit(univariate_time_series).decision_function(
            univariate_time_series
        )
        assert y_pred_1.shape == y_pred_2.shape
        assert not np.array_equal(y_pred_1, y_pred_2)

    def test_str(self):
        assert str(baselines.RandomDetector()) == "RandomDetector()"
        assert str(baselines.RandomDetector(42)) == "RandomDetector(seed=42)"
