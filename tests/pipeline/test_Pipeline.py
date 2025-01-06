import pytest

from dtaianomaly.preprocessing import Identity, StandardScaler, ChainedPreprocessor
from dtaianomaly.anomaly_detection import IsolationForest

from dtaianomaly.pipeline import Pipeline


class TestPipeline:

    def test_initialization(self):
        Pipeline(StandardScaler(), IsolationForest(15))

    def test_list_of_preprocessors(self):
        Pipeline([StandardScaler(), Identity()], IsolationForest(15))

    def test_no_preprocessors(self):
        with pytest.raises(ValueError):
            Pipeline([], IsolationForest(15))

    def test_invalid_preprocessor(self):
        with pytest.raises(TypeError):
            Pipeline('bonkers', IsolationForest(15))

    def test_invalid_list(self):
        with pytest.raises(TypeError):
            Pipeline([StandardScaler(), 'bonkers'], IsolationForest(15))

    def test_invalid_detector(self):
        with pytest.raises(TypeError):
            Pipeline(StandardScaler(), 'IsolationForest')

    def test_str(self):
        assert str(Pipeline(StandardScaler(), IsolationForest(15, 3))) == 'StandardScaler()->IsolationForest(window_size=15,stride=3)'
        assert (str(Pipeline(ChainedPreprocessor(Identity(), StandardScaler()), IsolationForest(15, 3)))
                == 'Identity()->StandardScaler()->IsolationForest(window_size=15,stride=3)')


