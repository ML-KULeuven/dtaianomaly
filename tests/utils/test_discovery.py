
import pytest

from dtaianomaly import anomaly_detection, data, evaluation, preprocessing, thresholding
from dtaianomaly.utils.discovery import all_classes

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
    anomaly_detection.MultivariateDetector,
    anomaly_detection.OneClassSupportVectorMachine,
    anomaly_detection.PrincipalComponentAnalysis,
    anomaly_detection.RobustPrincipalComponentAnalysis,
    anomaly_detection.DWT_MLEAD,
    anomaly_detection.AutoEncoder,
    anomaly_detection.MultilayerPerceptron,
    anomaly_detection.ConvolutionalNeuralNetwork,
    anomaly_detection.LongShortTermMemoryNetwork,
    anomaly_detection.Transformer,
    anomaly_detection.LocalPolynomialApproximation,
    anomaly_detection.ChronosAnomalyDetector,
    anomaly_detection.SpectralResidual,
    anomaly_detection.MOMENTAnomalyDetector,
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
    evaluation.AffiliationPrecision,
    evaluation.AffiliationRecall,
    evaluation.AffiliationFBeta,
    evaluation.RangeBasedPrecision,
    evaluation.RangeBasedRecall,
    evaluation.RangeBasedFBeta,
    evaluation.UCRScore
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
    evaluation.UCRScore
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
    evaluation.AffiliationPrecision,
    evaluation.AffiliationRecall,
    evaluation.AffiliationFBeta,
    evaluation.RangeBasedPrecision,
    evaluation.RangeBasedRecall,
    evaluation.RangeBasedFBeta,
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
everything = anomaly_detectors + data_loaders + metrics + preprocessors + thresholders


@pytest.mark.parametrize('return_names', [True, False])
@pytest.mark.parametrize('type_filter,expected', [
    (None, everything),
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
        discovered = all_classes(type_filter='anomaly-detector', return_names=False)
        assert len(discovered) == len(anomaly_detectors)
        for exp in anomaly_detectors:
            assert exp in discovered

    def test_type(self):
        discovered = all_classes(type_filter=anomaly_detection.BaseDetector, return_names=False)
        assert len(discovered) == len(anomaly_detectors)
        for exp in anomaly_detectors:
            assert exp in discovered

    def test_str_list(self):
        discovered = all_classes(type_filter=['anomaly-detector', 'data-loader'], return_names=False)
        assert len(discovered) == len(anomaly_detectors) + len(data_loaders)
        for exp in anomaly_detectors:
            assert exp in discovered
        for exp in data_loaders:
            assert exp in discovered

    def test_type_list(self):
        discovered = all_classes(type_filter=[anomaly_detection.BaseDetector, data.LazyDataLoader], return_names=False)
        assert len(discovered) == len(anomaly_detectors) + len(data_loaders)
        for exp in anomaly_detectors:
            assert exp in discovered
        for exp in data_loaders:
            assert exp in discovered

    def test_str_type_list(self):
        discovered = all_classes(type_filter=['anomaly-detector', data.LazyDataLoader], return_names=False)
        assert len(discovered) == len(anomaly_detectors) + len(data_loaders)
        for exp in anomaly_detectors:
            assert exp in discovered
        for exp in data_loaders:
            assert exp in discovered

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            all_classes(type_filter='invalid')


class TestExcludeType:

    def test_str(self):
        discovered = all_classes(exclude_types='anomaly-detector', return_names=False)
        assert len(discovered) == len(everything) - len(anomaly_detectors)
        for exp in everything:
            if exp not in anomaly_detectors:
                assert exp in discovered

    def test_type(self):
        discovered = all_classes(exclude_types=anomaly_detection.BaseDetector, return_names=False)
        assert len(discovered) == len(everything) - len(anomaly_detectors)
        for exp in everything:
            if exp not in anomaly_detectors:
                assert exp in discovered

    def test_str_list(self):
        discovered = all_classes(exclude_types=['anomaly-detector', 'data-loader'], return_names=False)
        assert len(discovered) == len(everything) - len(anomaly_detectors) - len(data_loaders)
        for exp in everything:
            if not (exp in anomaly_detectors or exp in data_loaders):
                assert exp in discovered

    def test_type_list(self):
        discovered = all_classes(exclude_types=[anomaly_detection.BaseDetector, data.LazyDataLoader], return_names=False)
        assert len(discovered) == len(everything) - len(anomaly_detectors) - len(data_loaders)
        for exp in everything:
            if not (exp in anomaly_detectors or exp in data_loaders):
                assert exp in discovered

    def test_str_type_list(self):
        discovered = all_classes(exclude_types=['anomaly-detector', data.LazyDataLoader], return_names=False)
        assert len(discovered) == len(everything) - len(anomaly_detectors) - len(data_loaders)
        for exp in everything:
            if not (exp in anomaly_detectors or exp in data_loaders):
                assert exp in discovered

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            all_classes(exclude_types='invalid')
