
from dtaianomaly.anomaly_detection import IsolationForest, LocalOutlierFactor
from dtaianomaly.evaluation import AreaUnderROC, Precision, ThresholdMetric
from dtaianomaly.thresholding import FixedCutoff, ContaminationRate
from dtaianomaly.preprocessing import Identity, ZNormalizer, ChainedPreprocessor
from dtaianomaly.workflow.utils import build_pipelines, convert_to_proba_metrics, convert_to_list


class TestBuildPipelines:

    def test(self):
        pipelines = build_pipelines(
            preprocessors=[Identity()],
            detectors=[IsolationForest(15)],
            metrics=[AreaUnderROC()]
        )
        assert len(pipelines) == 1

    def test_multiple_preprocessors(self):
        pipelines = build_pipelines(
            preprocessors=[Identity(), ZNormalizer()],
            detectors=[IsolationForest(15)],
            metrics=[AreaUnderROC()]
        )
        assert len(pipelines) == 2

    def test_multiple_detectors(self):
        pipelines = build_pipelines(
            preprocessors=[Identity()],
            detectors=[IsolationForest(15), LocalOutlierFactor(15)],
            metrics=[AreaUnderROC()]
        )
        assert len(pipelines) == 2

    def test_multiple_metrics(self):
        pipelines = build_pipelines(
            preprocessors=[Identity()],
            detectors=[IsolationForest(15)],
            metrics=[AreaUnderROC(), ThresholdMetric(ContaminationRate(0.05), Precision())]
        )
        assert len(pipelines) == 1

    def test_combinations(self):
        pipelines = build_pipelines(
            preprocessors=[Identity(), ZNormalizer()],
            detectors=[IsolationForest(15), LocalOutlierFactor(15)],
            metrics=[AreaUnderROC(), ThresholdMetric(ContaminationRate(0.05), Precision())]
        )
        assert len(pipelines) == 4

    def test_list_of_list_of_preprocessors(self):
        pipelines = build_pipelines(
            preprocessors=[Identity(), [ZNormalizer(), Identity()]],
            detectors=[IsolationForest(15), LocalOutlierFactor(15)],
            metrics=[AreaUnderROC(), ThresholdMetric(ContaminationRate(0.05), Precision())]
        )
        assert len(pipelines) == 4
        assert sum(isinstance(pipeline.pipeline.preprocessor, ChainedPreprocessor) for pipeline in pipelines) == 2


class TestConvertToProbaMetrics:

    def test(self):
        proba_metrics = convert_to_proba_metrics(
            metrics=[AreaUnderROC(), Precision()],
            thresholds=[ContaminationRate(0.05)],
        )
        assert len(proba_metrics) == 2
        assert sum(isinstance(proba_metric, ThresholdMetric) for proba_metric in proba_metrics) == 1

    def test_multiple_thresholds(self):
        proba_metrics = convert_to_proba_metrics(
            metrics=[AreaUnderROC(), Precision()],
            thresholds=[ContaminationRate(0.05), FixedCutoff(0.5)],
        )
        assert len(proba_metrics) == 3
        assert sum(isinstance(proba_metric, ThresholdMetric) for proba_metric in proba_metrics) == 2

    def test_no_binary_metric(self):
        proba_metrics = convert_to_proba_metrics(
            metrics=[AreaUnderROC()],
            thresholds=[ContaminationRate(0.05), FixedCutoff(0.5)],
        )
        assert len(proba_metrics) == 1
        assert sum(isinstance(proba_metric, ThresholdMetric) for proba_metric in proba_metrics) == 0

    def test_no_binary_metric_no_thresholds(self):
        proba_metrics = convert_to_proba_metrics(
            metrics=[AreaUnderROC()],
            thresholds=[],
        )
        assert len(proba_metrics) == 1
        assert sum(isinstance(proba_metric, ThresholdMetric) for proba_metric in proba_metrics) == 0


class TestConvertToList:

    def test_single_item(self):
        assert convert_to_list('5') == ['5']

    def test_list(self):
        assert convert_to_list(['5', '6']) == ['5', '6']

    def test_list_single_item(self):
        assert convert_to_list(['5']) == ['5']
