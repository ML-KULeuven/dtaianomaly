
import pytest

from dtaianomaly import anomaly_detection, data, evaluation, preprocessing, thresholding
from dtaianomaly.utils.discovery import all_classes, _TYPE_FILTERS

anomaly_detectors = [
    anomaly_detection.AlwaysNormal,
    anomaly_detection.AlwaysAnomalous,
    anomaly_detection.RandomDetector,
    anomaly_detection.ClusterBasedLocalOutlierFactor,
    anomaly_detection.CopulaBasedOutlierDetector,
    anomaly_detection.HistogramBasedOutlierScore,
    anomaly_detection.IsolationForest,
    anomaly_detection.KernelPrincipalComponentAnalysis,
    anomaly_detection.KMeansAnomalyDetector,
    anomaly_detection.KNearestNeighbors,
    anomaly_detection.KShapeAnomalyDetector,
    anomaly_detection.LocalOutlierFactor,
    anomaly_detection.MatrixProfileDetector,
    anomaly_detection.MedianMethod,
    anomaly_detection.OneClassSupportVectorMachine,
    anomaly_detection.PrincipalComponentAnalysis,
    anomaly_detection.RobustPrincipalComponentAnalysis,
]
data_loaders = [
    data.DemonstrationTimeSeriesLoader,
    data.UCRLoader
]
metrics = [
    evaluation.ThresholdMetric,
    evaluation.Precision,
    evaluation.Recall,
    evaluation.FBeta,
    evaluation.AreaUnderPR,
    evaluation.AreaUnderROC,
    evaluation.PointAdjustedPrecision,
    evaluation.PointAdjustedRecall,
    evaluation.PointAdjustedFBeta,
    evaluation.BestThresholdMetric,
    evaluation.RangeAreaUnderROC,
    evaluation.RangeAreaUnderPR,
    evaluation.VolumeUnderROC,
    evaluation.VolumeUnderPR,
    evaluation.EventWisePrecision,
    evaluation.EventWiseRecall,
    evaluation.EventWiseFBeta,
]
proba_metrics = [
    evaluation.ThresholdMetric,
    evaluation.AreaUnderPR,
    evaluation.AreaUnderROC,
    evaluation.BestThresholdMetric,
    evaluation.RangeAreaUnderROC,
    evaluation.RangeAreaUnderPR,
    evaluation.VolumeUnderROC,
    evaluation.VolumeUnderPR,
]
binary_metrics = [
    evaluation.Precision,
    evaluation.Recall,
    evaluation.FBeta,
    evaluation.PointAdjustedPrecision,
    evaluation.PointAdjustedRecall,
    evaluation.PointAdjustedFBeta,
    evaluation.EventWisePrecision,
    evaluation.EventWiseRecall,
    evaluation.EventWiseFBeta,
]
preprocessors = [
    preprocessing.Identity,
    preprocessing.ChainedPreprocessor,
    preprocessing.MinMaxScaler,
    preprocessing.StandardScaler,
    preprocessing.MovingAverage,
    preprocessing.ExponentialMovingAverage,
    preprocessing.SamplingRateUnderSampler,
    preprocessing.NbSamplesUnderSampler,
    preprocessing.Differencing,
    preprocessing.PiecewiseAggregateApproximation,
    preprocessing.RobustScaler,
]
thresholders = [
    thresholding.FixedCutoff,
    thresholding.ContaminationRate,
    thresholding.TopN
]


@pytest.mark.parametrize('return_names', [True, False])
@pytest.mark.parametrize('type_filter,expected', [
    (None, anomaly_detectors + data_loaders + metrics + preprocessors + thresholders),
    (anomaly_detection.BaseDetector, anomaly_detectors),
    (data.LazyDataLoader, data_loaders),
    (evaluation.Metric, metrics),
    (evaluation.ProbaMetric, proba_metrics),
    (evaluation.BinaryMetric, binary_metrics),
    (preprocessing.Preprocessor, preprocessors),
    (thresholding.Thresholding, thresholders),
])
class TestAllClasses:

    def test(self, type_filter, expected, return_names):
        discovered = all_classes(type_filter=type_filter, return_names=return_names)
        assert len(discovered) == len(expected)

        if return_names:
            discovered_dict = {t: n for n, t in discovered}
            for exp in expected:
                assert exp in discovered_dict
                assert exp.__name__ == discovered_dict[exp]

        else:
            for exp in expected:
                assert exp in discovered


class TestTypeFilter:

    def test_str(self):
        pass

    def test_type(self):
        pass

    def test_str_list(self):
        assert 0

    def test_type_list(self):
        assert 0

    def test_all_valid_strs(self):
        for key in _TYPE_FILTERS:
            all_classes(type_filter=key)


class TestExcludeType:

    def test_str(self):
        pass

    def test_type(self):
        pass

    def test_str_list(self):
        assert 0

    def test_type_list(self):
        assert 0

    def test_all_valid_strs(self):
        for key in _TYPE_FILTERS:
            all_classes(exclude_types=key)
