
import os
import pytest
import numpy as np

from dtaianomaly.anomaly_detection import BaseDetector, load_detector, baselines
from dtaianomaly import utils


class InvalidConstantDecisionFunctionForPredictProba(BaseDetector):

    def fit(self, X, y=None):
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.ones(X.shape[0]) * 50


class TestBaseDetector:

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
