
import numpy as np
import pytest

from dtaianomaly.preprocessing import Identity, StandardScaler, SamplingRateUnderSampler
from dtaianomaly.anomaly_detection import IsolationForest
from dtaianomaly.evaluation import AreaUnderROC, AreaUnderPR

from dtaianomaly.pipeline import EvaluationPipeline


class TestEvaluationPipeline:

    def test_initialization(self):
        pipeline = EvaluationPipeline(StandardScaler(), IsolationForest(15), AreaUnderPR())
        assert isinstance(pipeline.metrics, list)

    def test_list_of_metrics(self):
        EvaluationPipeline(StandardScaler(), IsolationForest(15), [AreaUnderPR(), AreaUnderROC()])

    def test_no_preprocessors(self):
        with pytest.raises(ValueError):
            EvaluationPipeline([], IsolationForest(15), AreaUnderPR())

    def test_invalid_preprocessor(self):
        with pytest.raises(TypeError):
            EvaluationPipeline('bonkers', IsolationForest(15), AreaUnderPR())

    def test_invalid_list(self):
        with pytest.raises(TypeError):
            EvaluationPipeline([StandardScaler(), 'bonkers'], IsolationForest(15), AreaUnderPR())

    def test_invalid_detector(self):
        with pytest.raises(TypeError):
            EvaluationPipeline(StandardScaler(), 'IsolationForest', AreaUnderPR())

    def test_eval_invalid_metrics(self):
        with pytest.raises(TypeError):
            EvaluationPipeline(StandardScaler(), IsolationForest(15), [AreaUnderROC(), 'AreaUnderPR'])
