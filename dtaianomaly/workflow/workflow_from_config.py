import inspect
import json
import os

import toml

from dtaianomaly import anomaly_detection, data, evaluation, preprocessing, thresholding
from dtaianomaly.workflow import Workflow


def workflow_from_config(path: str, max_size: int = 1000000):
    """
    Construct a Workflow instance based on a JSON or TOML file. The file is
    first parsed, and then interpreted to obtain a :py:class:`~dtaianomaly.workflow.Workflow`

    Parameters
    ----------
    path: str
        Path to the config file
    max_size: int, optional
        Maximal size of the config file in bytes. Defaults to 1 MB.

    Returns
    -------
    workflow: Workflow
        The parsed workflow from the given config file.

    Raises
    ------
    TypeError
        If the given path is not a string.
    FileNotFoundError
        If the given path does not correspond to an existing file.
    ValueError
        If the given path does not refer to a json or TOML file.
    """
    if not isinstance(path, str):
        raise TypeError("Path expects a string")
    if not os.path.exists(path):
        raise FileNotFoundError("The given path does not exist!")

    if path.endswith(".json"):
        with open(path, "r") as file:
            # Check file size
            file.seek(0, 2)
            file_size = file.tell()
            if file_size > max_size:
                raise ValueError(f"File size exceeds maximum size of {max_size} bytes")
            file.seek(0)

            # Parse actual JSON
            parsed_config = json.load(file)

    elif path.endswith(".toml"):
        with open(path, "r") as f:
            parsed_config = toml.load(f)

    else:
        raise ValueError("The given path should be a json or toml file!")

    return interpret_config(parsed_config)


def interpret_config(config: dict):
    """
    Actual parsing/interpretation logic

    All the different `_interpret_*` functions below check the config
    for the corresponding `dtaianomaly` objects. These functions should
    be extended when the full package is extended.

    Parameters
    ----------
    config: dict
        The config to parse

    Returns
    -------
    Workflow
        Containing all the components specified in the config
    """
    # Check the config file
    if not isinstance(config, dict):
        raise TypeError("Input should be a dictionary")

    # Interpret the data loaders
    if "dataloaders" not in config:
        raise ValueError("No `dataloaders` key in the config")
    if "data_root" in config:
        data_root = config["data_root"]
    else:
        data_root = ""
    dataloaders = interpret_dataloaders(config["dataloaders"], data_root=data_root)

    # Interpret the preprocessors
    preprocessors = (
        interpret_preprocessing(config["preprocessors"])
        if "preprocessors" in config
        else None
    )

    # Interpret the detectors
    if "detectors" not in config:
        raise ValueError("No `detectors` key in the config")
    detectors = interpret_detectors(config["detectors"])

    # Interpret the metrics
    if "metrics" not in config:
        raise ValueError("No `metrics` key in the config")
    metrics = interpret_metrics(config["metrics"])

    # Interpret the thresholds
    thresholds = (
        interpret_thresholds(config["thresholds"]) if "thresholds" in config else None
    )

    return Workflow(
        dataloaders=dataloaders,
        preprocessors=preprocessors,
        detectors=detectors,
        metrics=metrics,
        thresholds=thresholds,
        **interpret_additional_information(config),
    )


###################################################################
# THRESHOLDS
###################################################################


def interpret_thresholds(config):
    if isinstance(config, list):
        return [threshold_entry(entry) for entry in config]
    else:
        return [threshold_entry(config)]


def threshold_entry(entry):
    threshold_type = entry["type"]
    entry_without_type = {key: value for key, value in entry.items() if key != "type"}

    if threshold_type == "FixedCutoff":
        return thresholding.FixedCutoff(**entry_without_type)

    elif threshold_type == "ContaminationRate":
        return thresholding.ContaminationRate(**entry_without_type)

    elif threshold_type == "TopN":
        return thresholding.TopN(**entry_without_type)
    else:
        raise ValueError(f"Invalid threshold entry: {entry}")


###################################################################
# DATALOADERS
###################################################################


def interpret_dataloaders(config, data_root: str = ""):
    if isinstance(config, list):
        data_loaders = []
        for entry in config:
            new_loaders = data_entry(entry, data_root)
            if isinstance(new_loaders, list):
                data_loaders.extend(new_loaders)
            else:
                data_loaders.append(new_loaders)
        return data_loaders

    else:
        return [data_entry(config, data_root)]


def data_entry(entry, data_root: str = ""):
    data_type = entry["type"]
    entry_without_type = {key: value for key, value in entry.items() if key != "type"}

    if "path" in entry_without_type:
        entry_without_type["path"] = os.path.join(data_root, entry_without_type["path"])

    if data_type == "UCRLoader":
        return data.UCRLoader(**entry_without_type)

    elif data_type == "directory":
        if len(entry_without_type) != 2:
            raise ValueError(f"Incorrect number of items in entry: {entry}")
        if "path" not in entry:
            raise TypeError(f"Entry should have a path keyword: {entry}")
        if "base_type" not in entry:
            raise TypeError(f"Entry should have a base_type keyword: {entry}")

        if entry["base_type"] == "UCRLoader":
            base_type = data.UCRLoader
        else:
            raise ValueError(f"Invalid base type: {entry}")

        return data.from_directory(entry_without_type["path"], base_type)

    elif data_type == "DemonstrationTimeSeriesLoader":
        return data.DemonstrationTimeSeriesLoader(**entry_without_type)

    else:
        raise ValueError(f"Invalid data entry: {entry}")


###################################################################
# METRICS
###################################################################


def interpret_metrics(config):
    if isinstance(config, list):
        return [metric_entry(entry) for entry in config]
    else:
        return [metric_entry(config)]


def metric_entry(entry):
    metric_type = entry["type"]
    entry_without_type = {key: value for key, value in entry.items() if key != "type"}

    if metric_type == "Precision":
        if len(entry_without_type) > 0:
            raise TypeError(f"Too many parameters given for entry: {entry}")
        return evaluation.Precision()

    elif metric_type == "Recall":
        if len(entry_without_type) > 0:
            raise TypeError(f"Too many parameters given for entry: {entry}")
        return evaluation.Recall()

    elif metric_type == "FBeta":
        return evaluation.FBeta(**entry_without_type)

    elif metric_type == "AreaUnderROC":
        if len(entry_without_type) > 0:
            raise TypeError(f"Too many parameters given for entry: {entry}")
        return evaluation.AreaUnderROC()

    elif metric_type == "AreaUnderPR":
        if len(entry_without_type) > 0:
            raise TypeError(f"Too many parameters given for entry: {entry}")
        return evaluation.AreaUnderPR()

    elif metric_type == "PointAdjustedPrecision":
        if len(entry_without_type) > 0:
            raise TypeError(f"Too many parameters given for entry: {entry}")
        return evaluation.PointAdjustedPrecision()

    elif metric_type == "PointAdjustedRecall":
        if len(entry_without_type) > 0:
            raise TypeError(f"Too many parameters given for entry: {entry}")
        return evaluation.PointAdjustedRecall()

    elif metric_type == "PointAdjustedFBeta":
        return evaluation.PointAdjustedFBeta(**entry_without_type)

    elif metric_type == "ThresholdMetric":
        if len(entry_without_type) != 2:
            raise TypeError(
                f"BestThresholdMetric must have thresholder and metric as key: {entry}"
            )
        if "thresholder" not in entry:
            raise ValueError(
                f"BestThresholdMetric must have thresholder as key: {entry}"
            )
        if "metric" not in entry:
            raise ValueError(f"BestThresholdMetric must have metric as key: {entry}")
        return evaluation.ThresholdMetric(
            thresholder=threshold_entry(entry["thresholder"]),
            metric=metric_entry(entry["metric"]),
        )

    elif metric_type == "BestThresholdMetric":
        if "metric" not in entry:
            raise TypeError(f"BestThresholdMetric must have metric as key: {entry}")
        metric = metric_entry(entry["metric"])
        entry_without_metric = entry_without_type.copy()
        entry_without_metric.pop("metric")
        return evaluation.BestThresholdMetric(metric=metric, **entry_without_metric)

    elif metric_type == "RangeAreaUnderPR":
        return evaluation.RangeAreaUnderPR(**entry_without_type)

    elif metric_type == "RangeAreaUnderROC":
        return evaluation.RangeAreaUnderROC(**entry_without_type)

    elif metric_type == "VolumeUnderPR":
        return evaluation.VolumeUnderPR(**entry_without_type)

    elif metric_type == "VolumeUnderROC":
        return evaluation.VolumeUnderROC(**entry_without_type)

    elif metric_type == "EventWisePrecision":
        if len(entry_without_type) > 0:
            raise TypeError(f"Too many parameters given for entry: {entry}")
        return evaluation.EventWisePrecision()

    elif metric_type == "EventWiseRecall":
        if len(entry_without_type) > 0:
            raise TypeError(f"Too many parameters given for entry: {entry}")
        return evaluation.EventWiseRecall()

    elif metric_type == "EventWiseFBeta":
        return evaluation.EventWiseFBeta(**entry_without_type)

    else:
        raise ValueError(f"Invalid metric entry: {entry}")


###################################################################
# DETECTORS
###################################################################


def interpret_detectors(config):
    if isinstance(config, list):
        return [detector_entry(entry) for entry in config]
    else:
        return [detector_entry(config)]


def detector_entry(entry):
    detector_type = entry["type"]
    entry_without_type = {key: value for key, value in entry.items() if key != "type"}

    if detector_type == "AlwaysNormal":
        if len(entry_without_type) > 0:
            raise TypeError(f"Too many parameters given for entry: {entry}")
        return anomaly_detection.baselines.AlwaysNormal()

    elif detector_type == "AlwaysAnomalous":
        if len(entry_without_type) > 0:
            raise TypeError(f"Too many parameters given for entry: {entry}")
        return anomaly_detection.baselines.AlwaysAnomalous()

    elif detector_type == "RandomDetector":
        return anomaly_detection.baselines.RandomDetector(**entry_without_type)

    elif detector_type == "MatrixProfileDetector":
        return anomaly_detection.MatrixProfileDetector(**entry_without_type)

    elif detector_type == "IsolationForest":
        return anomaly_detection.IsolationForest(**entry_without_type)

    elif detector_type == "LocalOutlierFactor":
        return anomaly_detection.LocalOutlierFactor(**entry_without_type)

    elif detector_type == "MedianMethod":
        return anomaly_detection.MedianMethod(**entry_without_type)

    elif detector_type == "KNearestNeighbors":
        return anomaly_detection.KNearestNeighbors(**entry_without_type)

    elif detector_type == "HistogramBasedOutlierScore":
        return anomaly_detection.HistogramBasedOutlierScore(**entry_without_type)

    elif detector_type == "PrincipalComponentAnalysis":
        return anomaly_detection.PrincipalComponentAnalysis(**entry_without_type)

    elif detector_type == "KernelPrincipalComponentAnalysis":
        return anomaly_detection.KernelPrincipalComponentAnalysis(**entry_without_type)

    elif detector_type == "RobustPrincipalComponentAnalysis":
        return anomaly_detection.RobustPrincipalComponentAnalysis(**entry_without_type)

    elif detector_type == "OneClassSupportVectorMachine":
        return anomaly_detection.OneClassSupportVectorMachine(**entry_without_type)

    elif detector_type == "ClusterBasedLocalOutlierFactor":
        return anomaly_detection.ClusterBasedLocalOutlierFactor(**entry_without_type)

    elif detector_type == "KMeansAnomalyDetector":
        return anomaly_detection.KMeansAnomalyDetector(**entry_without_type)

    elif detector_type == "CopulaBasedOutlierDetector":
        return anomaly_detection.CopulaBasedOutlierDetector(**entry_without_type)

    elif detector_type == "KShapeAnomalyDetector":
        return anomaly_detection.KShapeAnomalyDetector(**entry_without_type)

    elif detector_type == "MultivariateDetector":
        if "detector" not in entry:
            raise TypeError(f"MultivariateDetector must have detector as key: {entry}")
        detector = detector_entry(entry["detector"])
        entry_without_detector = entry_without_type.copy()
        entry_without_detector.pop("detector")
        return anomaly_detection.MultivariateDetector(
            detector, **entry_without_detector
        )

    elif detector_type == "DWT_MLEAD":
        return anomaly_detection.DWT_MLEAD(**entry_without_type)

    else:
        raise ValueError(f"Invalid detector entry: {entry}")


###################################################################
# PREPROCESSING
###################################################################


def interpret_preprocessing(config):
    if isinstance(config, list):
        return [preprocessing_entry(entry) for entry in config]
    else:
        return [preprocessing_entry(config)]


def preprocessing_entry(entry):
    processing_type = entry["type"]
    entry_without_type = {key: value for key, value in entry.items() if key != "type"}

    if processing_type == "Identity":
        if len(entry_without_type) > 0:
            raise TypeError(f"Too many parameters given for entry: {entry}")
        return preprocessing.Identity()

    elif processing_type == "MinMaxScaler":
        if len(entry_without_type) > 0:
            raise TypeError(f"Too many parameters given for entry: {entry}")
        return preprocessing.MinMaxScaler()

    elif processing_type == "StandardScaler":
        return preprocessing.StandardScaler(**entry_without_type)

    elif processing_type == "MovingAverage":
        return preprocessing.MovingAverage(**entry_without_type)

    elif processing_type == "ExponentialMovingAverage":
        return preprocessing.ExponentialMovingAverage(**entry_without_type)

    elif processing_type == "NbSamplesUnderSampler":
        return preprocessing.NbSamplesUnderSampler(**entry_without_type)

    elif processing_type == "SamplingRateUnderSampler":
        return preprocessing.SamplingRateUnderSampler(**entry_without_type)

    elif processing_type == "Differencing":
        return preprocessing.Differencing(**entry_without_type)

    elif processing_type == "PiecewiseAggregateApproximation":
        return preprocessing.PiecewiseAggregateApproximation(**entry_without_type)

    elif processing_type == "RobustScaler":
        return preprocessing.RobustScaler(**entry_without_type)

    elif processing_type == "ChainedPreprocessor":
        if len(entry_without_type) != 1:
            raise TypeError(
                f"ChainedPreprocessor must have base_preprocessors as key: {entry}"
            )
        if "base_preprocessors" not in entry:
            raise TypeError(
                f"ChainedPreprocessor must have base_preprocessors as key: {entry}"
            )
        if not isinstance(entry["base_preprocessors"], list):
            raise ValueError(
                "A chained preprocessor should have a list as base_preprocessors!"
            )
        return preprocessing.ChainedPreprocessor(
            [preprocessing_entry(e) for e in entry["base_preprocessors"]]
        )

    else:
        raise ValueError(f"Invalid preprocessing config: {entry}")


###################################################################
# ADDITIONAL INFORMATION
###################################################################


def interpret_additional_information(config):
    return {
        argument: config[argument]
        for argument in inspect.signature(Workflow.__init__).parameters.keys()
        if argument in config
        and argument
        not in [
            "self",
            "dataloaders",
            "metrics",
            "detectors",
            "preprocessors",
            "thresholds",
        ]
    }
