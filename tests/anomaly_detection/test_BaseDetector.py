
import os
import pytest
import numpy as np

from dtaianomaly.anomaly_detection import BaseDetector, load_detector, baselines, Supervision
from dtaianomaly import utils


class InvalidConstantDecisionFunctionForPredictProba(BaseDetector):

    def __init__(self):
        super().__init__(Supervision.UNSUPERVISED)

    def fit(self, X, y=None):
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.ones(X.shape[0]) * 50


class NoDefinedSupervisionDetector(BaseDetector):

    def fit(self, X, y=None):
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.ones(X.shape[0])


class TestBaseDetector:

    @pytest.mark.parametrize('supervision', Supervision)
    def test_valid_supervision(self, supervision: Supervision):
        detector = NoDefinedSupervisionDetector(supervision)
        assert detector.supervision == supervision

    def test_invalid_supervision(self):
        with pytest.raises(TypeError):
            NoDefinedSupervisionDetector('supervision.SUPERVISED')

    def test_proba(self):
        data = np.random.standard_normal((50,))
        detector = baselines.RandomDetector()
        probas = detector.predict_proba(data)
        assert np.all((0.0 <= probas) & (probas <= 1.0))

    def test_proba_always_normal(self):
        data = np.random.standard_normal((50,))
        detector = baselines.AlwaysNormal()
        probas = detector.predict_proba(data)
        assert np.all(probas == 0.0)

    def test_proba_always_anomalous(self):
        data = np.random.standard_normal((50,))
        detector = baselines.AlwaysAnomalous()
        probas = detector.predict_proba(data)
        assert np.all(probas == 1.0)

    def test_proba_invalid_constant_decision_function(self):
        data = np.random.standard_normal((50,))
        detector = InvalidConstantDecisionFunctionForPredictProba()
        with pytest.raises(ValueError):
            detector.predict_proba(data)

    def test_proba_invalid(self):
        invalid_data = None
        detector = baselines.RandomDetector()
        assert not utils.is_valid_array_like(invalid_data)
        with pytest.raises(ValueError):
            detector.predict_proba(invalid_data)

    def test_save_invalid_path(self, tmp_path):
        detector = baselines.RandomDetector()
        detector.save(tmp_path / 'some' / 'invalid' / 'directory' / 'testing')
        assert os.path.exists(tmp_path / 'some' / 'invalid' / 'directory')

    def test_save_and_load(self, tmp_path):

        # Save the detector
        detector = baselines.RandomDetector()
        detector.save(tmp_path / 'testing')

        # Load the detector
        loaded_detector = load_detector(tmp_path / 'testing.dtai')

        # Check if the original detector and the loaded detector have the same properties
        assert detector.__dict__ == loaded_detector.__dict__

        # The loaded detector can make a prediction
        data = np.random.standard_normal((50,))
        _ = loaded_detector.predict_proba(data)

    def test_str(self):
        assert str(baselines.RandomDetector()) == 'RandomDetector()'
        assert str(baselines.AlwaysNormal()) == 'AlwaysNormal()'


class TestConfidence:

    def test_predict_confidence(self, univariate_time_series):
        X_train = univariate_time_series[:int(univariate_time_series.shape[0]*0.3)]
        X_test = univariate_time_series[int(univariate_time_series.shape[0]*0.3):]

        detector = baselines.RandomDetector().fit(X_train)
        confidence = detector.predict_confidence(X_test, X_train)
        assert confidence.shape[0] == X_test.shape[0]
        assert len(confidence.shape) == 1

    def test_predict_confidence_multivariate(self, multivariate_time_series):
        X_train = multivariate_time_series[:int(multivariate_time_series.shape[0]*0.3), :]
        X_test = multivariate_time_series[int(multivariate_time_series.shape[0]*0.3):, :]

        detector = baselines.RandomDetector().fit(X_train)
        confidence = detector.predict_confidence(X_test, X_train)
        assert confidence.shape[0] == X_test.shape[0]
        assert len(confidence.shape) == 1

    def test_predict_confidence_no_train_data(self, univariate_time_series):
        detector = baselines.RandomDetector().fit(univariate_time_series)
        confidence = detector.predict_confidence(univariate_time_series)
        assert confidence.shape[0] == univariate_time_series.shape[0]
        assert len(confidence.shape) == 1

    def test_predict_confidence_decision_scores_given(self, univariate_time_series):
        detector = baselines.RandomDetector(seed=42).fit(univariate_time_series)
        decision_scores = detector.decision_function(univariate_time_series)
        confidence = detector.predict_confidence(decision_scores, decision_scores_given=True)
        assert confidence.shape[0] == univariate_time_series.shape[0]
        assert len(confidence.shape) == 1

        confidence_other = detector.predict_confidence(univariate_time_series)
        assert np.array_equal(confidence, confidence_other)

    def test_predict_confidence_decision_scores_train_and_test_given(self, univariate_time_series):
        X_train = univariate_time_series[:int(univariate_time_series.shape[0]*0.3)]
        X_test = univariate_time_series[int(univariate_time_series.shape[0]*0.3):]
        detector = baselines.RandomDetector(seed=42).fit(X_train)
        decision_scores = detector.decision_function(X_test)
        decision_scores_train = detector.decision_function(X_train)
        confidence = detector.predict_confidence(decision_scores, decision_scores_train, decision_scores_given=True)
        assert confidence.shape[0] == X_test.shape[0]
        assert len(confidence.shape) == 1

        confidence_other = detector.predict_confidence(X_test, X_train)
        assert np.array_equal(confidence, confidence_other)

    def test_predict_confidence_invalid_decision_scores_given(self, univariate_time_series):
        univariate_time_series = univariate_time_series.reshape(-1, 1)  # To make sure it has two dimensions
        assert len(univariate_time_series.shape) > 1

        detector = baselines.RandomDetector().fit(univariate_time_series)
        with pytest.raises(ValueError):
            detector.predict_confidence(univariate_time_series, decision_scores_given=True)

    def test_predict_confidence_invalid_decision_scores_train_given(self, univariate_time_series):
        univariate_time_series = univariate_time_series.reshape(-1, 1)  # To make sure it has two dimensions
        assert len(univariate_time_series.shape) > 1

        X_train = univariate_time_series[:int(univariate_time_series.shape[0]*0.3), :]
        X_test = univariate_time_series[int(univariate_time_series.shape[0]*0.3):, :]

        detector = baselines.RandomDetector().fit(X_train)
        decision_scores = detector.decision_function(X_test)

        with pytest.raises(ValueError):
            detector.predict_confidence(decision_scores, X_train, decision_scores_given=True)

    def test_repeatability(self, univariate_time_series):
        detector = baselines.RandomDetector(seed=42).fit(univariate_time_series)
        confidence1 = detector.predict_confidence(univariate_time_series)
        confidence2 = detector.predict_confidence(univariate_time_series)
        print((confidence1 != confidence2).sum())
        assert np.array_equal(confidence1, confidence2)
