
import numpy as np
import pytest

from dtaianomaly.preprocessing import Identity, ZNormalizer, SamplingRateUnderSampler
from dtaianomaly.anomaly_detection import IsolationForest
from dtaianomaly.evaluation import AreaUnderROC, AreaUnderPR

from dtaianomaly.pipeline import EvaluationPipeline


class TestEvaluationPipeline:

    def test_initialization(self):
        pipeline = EvaluationPipeline(ZNormalizer(), IsolationForest(15), AreaUnderPR())
        assert isinstance(pipeline.metrics, list)

    def test_list_of_metrics(self):
        EvaluationPipeline(ZNormalizer(), IsolationForest(15), [AreaUnderPR(), AreaUnderROC()])

    def test_no_preprocessors(self):
        with pytest.raises(ValueError):
            EvaluationPipeline([], IsolationForest(15), AreaUnderPR())

    def test_invalid_preprocessor(self):
        with pytest.raises(TypeError):
            EvaluationPipeline('bonkers', IsolationForest(15), AreaUnderPR())

    def test_invalid_list(self):
        with pytest.raises(TypeError):
            EvaluationPipeline([ZNormalizer(), 'bonkers'], IsolationForest(15), AreaUnderPR())

    def test_invalid_detector(self):
        with pytest.raises(TypeError):
            EvaluationPipeline(ZNormalizer(), 'IsolationForest', AreaUnderPR())

    def test_eval_invalid_metrics(self):
        with pytest.raises(TypeError):
            EvaluationPipeline(ZNormalizer(), IsolationForest(15), [AreaUnderROC(), 'AreaUnderPR'])

    def test_run_invalid_X_test(self):
        pipeline = EvaluationPipeline(Identity(), IsolationForest(15), AreaUnderROC())
        with pytest.raises(ValueError):
            pipeline.run(['3', '5', '7'], np.array([0, 1, 0]), np.array([1, 2, 3]), None)

    def test_run_invalid_y_test(self, univariate_time_series):
        pipeline = EvaluationPipeline(Identity(), IsolationForest(15), AreaUnderROC())
        with pytest.raises(ValueError):
            pipeline.run(np.array([3, 5, 7]), ['0', '1', '0'], np.array([1, 2, 3]), None)

    def test_run_none_y_test(self, univariate_time_series):
        pipeline = EvaluationPipeline(Identity(), IsolationForest(15), AreaUnderROC())
        with pytest.raises(ValueError):
            pipeline.run(np.array([3, 5, 7]), None, np.array([1, 2, 3]), None)

    def test_run_invalid_X_train(self):
        pipeline = EvaluationPipeline(Identity(), IsolationForest(15), AreaUnderROC())
        with pytest.raises(ValueError):
            pipeline.run([3, 5, 7], np.array([0, 1, 0]), np.array(['1', '2', '3']), None)

    def test_run_invalid_y_train(self, univariate_time_series):
        pipeline = EvaluationPipeline(Identity(), IsolationForest(15), AreaUnderROC())
        with pytest.raises(ValueError):
            pipeline.run(np.array([3, 5, 7]), [0, 1, 0], np.array([1, 2, 3]), np.array(['0', '1', '0']))

    def test_run(self, univariate_time_series):
        pipeline = EvaluationPipeline(ZNormalizer(), IsolationForest(15), [AreaUnderROC(), AreaUnderPR()])
        y = np.random.choice([0, 1], size=univariate_time_series.shape[0], replace=True)
        results = pipeline.run(univariate_time_series, y, univariate_time_series, None)
        assert len(results) == 2
        assert 'AreaUnderROC()' in results
        assert 'AreaUnderPR()' in results

    def test_run_multivariate(self, multivariate_time_series):
        pipeline = EvaluationPipeline(ZNormalizer(), IsolationForest(15), [AreaUnderROC(), AreaUnderPR()])
        y = np.random.choice([0, 1], size=multivariate_time_series.shape[0], replace=True)
        results = pipeline.run(multivariate_time_series, y, multivariate_time_series, y)
        assert len(results) == 2
        assert 'AreaUnderROC()' in results
        assert 'AreaUnderPR()' in results

    def test_shorter_y(self, univariate_time_series):
        pipeline = EvaluationPipeline(SamplingRateUnderSampler(5), IsolationForest(15), [AreaUnderROC(), AreaUnderPR()])
        y = np.random.choice([0, 1], size=univariate_time_series.shape[0], replace=True)
        results = pipeline.run(univariate_time_series, y, univariate_time_series, None)
        assert len(results) == 2
        assert 'AreaUnderROC()' in results
        assert 'AreaUnderPR()' in results
