from dtaianomaly.evaluation import AreaUnderROC, Precision, ThresholdMetric
from dtaianomaly.thresholding import ContaminationRateThreshold, FixedCutoffThreshold
from dtaianomaly.workflow.utils import convert_to_list, convert_to_proba_metrics


class TestConvertToProbaMetrics:

    def test(self):
        proba_metrics = convert_to_proba_metrics(
            metrics=[AreaUnderROC(), Precision()],
            thresholds=[ContaminationRateThreshold(0.05)],
        )
        assert len(proba_metrics) == 2
        assert (
            sum(
                isinstance(proba_metric, ThresholdMetric)
                for proba_metric in proba_metrics
            )
            == 1
        )

    def test_multiple_thresholds(self):
        proba_metrics = convert_to_proba_metrics(
            metrics=[AreaUnderROC(), Precision()],
            thresholds=[ContaminationRateThreshold(0.05), FixedCutoffThreshold(0.5)],
        )
        assert len(proba_metrics) == 3
        assert (
            sum(
                isinstance(proba_metric, ThresholdMetric)
                for proba_metric in proba_metrics
            )
            == 2
        )

    def test_no_binary_metric(self):
        proba_metrics = convert_to_proba_metrics(
            metrics=[AreaUnderROC()],
            thresholds=[ContaminationRateThreshold(0.05), FixedCutoffThreshold(0.5)],
        )
        assert len(proba_metrics) == 1
        assert (
            sum(
                isinstance(proba_metric, ThresholdMetric)
                for proba_metric in proba_metrics
            )
            == 0
        )

    def test_no_binary_metric_no_thresholds(self):
        proba_metrics = convert_to_proba_metrics(
            metrics=[AreaUnderROC()],
            thresholds=[],
        )
        assert len(proba_metrics) == 1
        assert (
            sum(
                isinstance(proba_metric, ThresholdMetric)
                for proba_metric in proba_metrics
            )
            == 0
        )


class TestConvertToList:

    def test_single_item(self):
        assert convert_to_list("5") == ["5"]

    def test_list(self):
        assert convert_to_list(["5", "6"]) == ["5", "6"]

    def test_list_single_item(self):
        assert convert_to_list(["5"]) == ["5"]
