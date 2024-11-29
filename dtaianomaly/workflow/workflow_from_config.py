
import os
import json
from itertools import chain

from dtaianomaly.workflow import Workflow
from dtaianomaly import preprocessing, anomaly_detection, evaluation, thresholding, data


def workflow_from_config(path: str, max_size: int = 1000000):
    """
    Construct a Workflow instance based on a JSON file. The file is
    first parsed, and then interpreted to obtain a :py:class:`~dtaianomaly.workflow.Workflow`

    Parameters
    ----------
    path: str
        Path to the config file in JSON format
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
        If the given path does not refer to a json file.
    """
    if not isinstance(path, str):
        raise TypeError("Path expects a string")
    if not os.path.exists(path):
        raise FileNotFoundError('The given path does not exist!')
    if not path.endswith('.json'):
        raise ValueError('The given path should be a json file!')

    with open(path, 'r') as file:
        # Check file size
        file.seek(0, 2)
        file_size = file.tell()
        if file_size > max_size:
            raise ValueError(f"File size exceeds maximum size of {max_size} bytes")
        file.seek(0)

        # Parse actual JSON
        parsed_config = json.load(file)

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
    if not isinstance(config, dict):
        raise TypeError("Input should be a dictionary")

    return Workflow(
        dataloaders=interpret_dataloaders(config),
        preprocessors=interpret_preprocessing(config),
        detectors=interpret_detectors(config),
        metrics=interpret_metrics(config),
        thresholds=interpret_thresholds(config),
        **interpret_additional_information(config)
    )


###################################################################
# THRESHOLDS
###################################################################

def interpret_thresholds(config):
    if 'thresholds' not in config:
        return None

    threshold_config = config['thresholds']
    if isinstance(threshold_config, list):
        return [threshold_entry(entry) for entry in threshold_config]
    else:
        return [threshold_entry(threshold_config),]


def threshold_entry(entry):
    threshold_type = entry['type']
    entry_without_type = {key: value for key, value in entry.items() if key != 'type'}

    if threshold_type == 'FixedCutoff':
        return thresholding.FixedCutoff(**entry_without_type)

    elif threshold_type == 'ContaminationRate':
        return thresholding.ContaminationRate(**entry_without_type)

    elif threshold_type == 'TopN':
        return thresholding.TopN(**entry_without_type)
    else:
        raise ValueError(f'Invalid threshold entry: {entry}')


###################################################################
# DATALOADERS
###################################################################

def interpret_dataloaders(config):
    if 'dataloaders' not in config:
        raise ValueError('No `dataloaders` key in the config')

    data_config = config['dataloaders']
    if isinstance(data_config, list):
        data_loaders = []
        for entry in data_config:
            new_loaders = data_entry(entry)
            if isinstance(new_loaders, list):
                data_loaders.extend(new_loaders)
            else:
                data_loaders.append(new_loaders)
        return data_loaders

    else:
        return [data_entry(data_config)]


def data_entry(entry):
    data_type = entry['type']
    entry_without_type = {key: value for key, value in entry.items() if key != 'type'}

    if data_type == 'UCRLoader':
        return data.UCRLoader(**entry_without_type)

    elif data_type == 'directory':
        if len(entry_without_type) != 2:
            raise ValueError(f'Incorrect number of items in entry: {entry}')
        if 'path' not in entry:
            raise TypeError(f'Entry should have a path keyword: {entry}')
        if 'base_type' not in entry:
            raise TypeError(f'Entry should have a base_type keyword: {entry}')

        if entry['base_type'] == 'UCRLoader':
            base_type = data.UCRLoader
        else:
            raise ValueError(f'Invalid base type: {entry}')

        return data.from_directory(entry['path'], base_type)

    else:
        raise ValueError(f'Invalid data entry: {entry}')


###################################################################
# METRICS
###################################################################

def interpret_metrics(config):
    if 'metrics' not in config:
        raise ValueError('No `metrics` key in the config')

    metric_config = config['metrics']
    if isinstance(metric_config, list):
        return [metric_entry(entry) for entry in metric_config]
    else:
        return [metric_entry(metric_config)]


def metric_entry(entry):
    metric_type = entry['type']
    entry_without_type = {key: value for key, value in entry.items() if key != 'type'}

    if metric_type == 'Precision':
        if len(entry_without_type) > 0:
            raise TypeError(f'Too many parameters given for entry: {entry}')
        return evaluation.Precision()

    elif metric_type == 'Recall':
        if len(entry_without_type) > 0:
            raise TypeError(f'Too many parameters given for entry: {entry}')
        return evaluation.Recall()

    elif metric_type == 'FBeta':
        return evaluation.FBeta(**entry_without_type)

    elif metric_type == 'AreaUnderROC':
        if len(entry_without_type) > 0:
            raise TypeError(f'Too many parameters given for entry: {entry}')
        return evaluation.AreaUnderROC()

    elif metric_type == 'AreaUnderPR':
        if len(entry_without_type) > 0:
            raise TypeError(f'Too many parameters given for entry: {entry}')
        return evaluation.AreaUnderPR()

    elif metric_type == 'PointAdjustedPrecision':
        if len(entry_without_type) > 0:
            raise TypeError(f'Too many parameters given for entry: {entry}')
        return evaluation.PointAdjustedPrecision()

    elif metric_type == 'PointAdjustedRecall':
        if len(entry_without_type) > 0:
            raise TypeError(f'Too many parameters given for entry: {entry}')
        return evaluation.PointAdjustedRecall()

    elif metric_type == 'PointAdjustedFBeta':
        return evaluation.PointAdjustedFBeta(**entry_without_type)

    elif metric_type == 'ThresholdMetric':
        if len(entry_without_type) != 2:
            raise TypeError(f'BestThresholdMetric must have thresholder and metric as key: {entry}')
        if 'thresholder' not in entry:
            raise ValueError(f'BestThresholdMetric must have thresholder as key: {entry}')
        if 'metric' not in entry:
            raise ValueError(f'BestThresholdMetric must have metric as key: {entry}')
        return evaluation.ThresholdMetric(thresholder=threshold_entry(entry['thresholder']), metric=metric_entry(entry['metric']))

    elif metric_type == 'BestThresholdMetric':
        if len(entry_without_type) != 1:
            raise TypeError(f'BestThresholdMetric must have metric as key: {entry}')
        if 'metric' not in entry:
            raise ValueError(f'BestThresholdMetric must have metric as key: {entry}')
        return evaluation.BestThresholdMetric(metric=metric_entry(entry['metric']))

    else:
        raise ValueError(f'Invalid metric entry: {entry}')


###################################################################
# DETECTORS
###################################################################

def interpret_detectors(config):
    if 'detectors' not in config:
        raise ValueError('No `detectors` key in the config')

    detector_config = config['detectors']
    if isinstance(detector_config, list):
        return [detector_entry(entry) for entry in detector_config]
    else:
        return [detector_entry(detector_config)]


def detector_entry(entry):
    detector_type = entry['type']
    entry_without_type = {key: value for key, value in entry.items() if key != 'type'}

    if detector_type == 'AlwaysNormal':
        if len(entry_without_type) > 0:
            raise TypeError(f'Too many parameters given for entry: {entry}')
        return anomaly_detection.baselines.AlwaysNormal()

    elif detector_type == 'AlwaysAnomalous':
        if len(entry_without_type) > 0:
            raise TypeError(f'Too many parameters given for entry: {entry}')
        return anomaly_detection.baselines.AlwaysAnomalous()

    elif detector_type == 'RandomDetector':
        return anomaly_detection.baselines.RandomDetector(**entry_without_type)

    elif detector_type == 'MatrixProfileDetector':
        return anomaly_detection.MatrixProfileDetector(**entry_without_type)

    elif detector_type == 'IsolationForest':
        return anomaly_detection.IsolationForest(**entry_without_type)

    elif detector_type == 'LocalOutlierFactor':
        return anomaly_detection.LocalOutlierFactor(**entry_without_type)

    elif detector_type == 'MedianMethod':
        return anomaly_detection.MedianMethod(**entry_without_type)

    elif detector_type == 'KNearestNeighbors':
        return anomaly_detection.KNearestNeighbors(**entry_without_type)

    elif detector_type == 'HistogramBasedOutlierScore':
        return anomaly_detection.HistogramBasedOutlierScore(**entry_without_type)

    elif detector_type == 'PrincipalComponentAnalysis':
        return anomaly_detection.PrincipalComponentAnalysis(**entry_without_type)

    elif detector_type == 'KernelPrincipalComponentAnalysis':
        return anomaly_detection.KernelPrincipalComponentAnalysis(**entry_without_type)

    elif detector_type == 'RobustPrincipalComponentAnalysis':
        return anomaly_detection.RobustPrincipalComponentAnalysis(**entry_without_type)

    else:
        raise ValueError(f'Invalid detector entry: {entry}')


###################################################################
# PREPROCESSING
###################################################################

def interpret_preprocessing(config):
    if "preprocessors" not in config:
        return None

    preprocessing_config = config["preprocessors"]

    if isinstance(preprocessing_config, list):
        return [preprocessing_entry(entry) for entry in preprocessing_config]
    else:
        return [preprocessing_entry(preprocessing_config)]


def preprocessing_entry(entry):
    processing_type = entry['type']
    entry_without_type = {key: value for key, value in entry.items() if key != 'type'}

    if processing_type == 'Identity':
        if len(entry_without_type) > 0:
            raise TypeError(f'Too many parameters given for entry: {entry}')
        return preprocessing.Identity()

    elif processing_type == 'MinMaxScaler':
        if len(entry_without_type) > 0:
            raise TypeError(f'Too many parameters given for entry: {entry}')
        return preprocessing.MinMaxScaler()

    elif processing_type == 'ZNormalizer':
        if len(entry_without_type) > 0:
            raise TypeError(f'Too many parameters given for entry: {entry}')
        return preprocessing.ZNormalizer()

    elif processing_type == 'MovingAverage':
        return preprocessing.MovingAverage(**entry_without_type)

    elif processing_type == 'ExponentialMovingAverage':
        return preprocessing.ExponentialMovingAverage(**entry_without_type)

    elif processing_type == 'NbSamplesUnderSampler':
        return preprocessing.NbSamplesUnderSampler(**entry_without_type)

    elif processing_type == 'SamplingRateUnderSampler':
        return preprocessing.SamplingRateUnderSampler(**entry_without_type)

    elif processing_type == 'Differencing':
        return preprocessing.Differencing(**entry_without_type)

    elif processing_type == 'PiecewiseAggregateApproximation':
        return preprocessing.PiecewiseAggregateApproximation(**entry_without_type)

    elif processing_type == 'ChainedPreprocessor':
        if len(entry_without_type) != 1:
            raise TypeError(f'ChainedPreprocessor must have base_preprocessors as key: {entry}')
        if 'base_preprocessors' not in entry:
            raise ValueError(f'ChainedPreprocessor must have base_preprocessors as key: {entry}')
        if not isinstance(entry['base_preprocessors'], list):
            raise ValueError('A chained preprocessor should have a list as base_preprocessors!')
        return preprocessing.ChainedPreprocessor([preprocessing_entry(e) for e in entry['base_preprocessors']])

    else:
        raise ValueError(f'Invalid preprocessing config: {entry}')


###################################################################
# ADDITIONAL INFORMATION
###################################################################

def interpret_additional_information(config):
    return {
        feature: config[feature]
        for feature in ['n_jobs', 'trace_memory']
        if feature in config
    }
