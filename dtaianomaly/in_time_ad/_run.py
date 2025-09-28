import pathlib
import sys
import warnings

import torch
from streamlit.web import cli as stcli

from dtaianomaly.anomaly_detection import BaseDetector
from dtaianomaly.data import LazyDataLoader
from dtaianomaly.evaluation import Metric
from dtaianomaly.in_time_ad._CustomDetectorVisualizer import CustomDetectorVisualizer
from dtaianomaly.utils import convert_to_list

torch.classes.__path__ = []  # To avoid torch-warning

__all__ = ["run"]


def run(
    configuration_path: str = None,
    custom_data_loaders: type[LazyDataLoader] | list[type[LazyDataLoader]] = None,
    custom_anomaly_detectors: type[BaseDetector] | list[type[BaseDetector]] = None,
    custom_metrics: type[Metric] | list[type[Metric]] = None,
    custom_visualizers: (
        type[CustomDetectorVisualizer] | list[type[CustomDetectorVisualizer]]
    ) = None,
):
    """
    Run InTimeAD.

    Start up the demonstrator for ``dtaianomaly``. This function will start a web-application
    on your local host, to which you can navigate in order to see InTimeAD.

    Parameters
    ----------
    configuration_path : str, default=None
        The path to the configuration file for the demonstrator. The configuration file
        must be in a json format.
    custom_data_loaders : LazyDataLoader object or list of LazyDataLoader objects, default=None
        Additional data loaders which must be available within the demonstrator.
    custom_anomaly_detectors : BaseDetector object or list of BaseDetector objects, default=None
        Additional anomaly detectors which must be available within the demonstrator.
    custom_metrics : Metric object or list of Metric objects, default=None
        Additional evaluation metrics which must be available within the demonstrator.
    custom_visualizers : CustomDetectorVisualizer object or list of CustomDetectorVisualizer objects, default=None
        Additional custom visualizations for certain anomaly detectors.
    """
    # Run the applications
    sys.argv = [
        "streamlit",
        "run",
        str(pathlib.Path(__file__).parent / "_app.py"),
        configuration_path or "default",
        str(
            _custom_model_config(
                custom_data_loaders=custom_data_loaders,
                custom_anomaly_detectors=custom_anomaly_detectors,
                custom_metrics=custom_metrics,
                custom_visualizers=custom_visualizers,
            )
        ),
    ]
    sys.exit(stcli.main())


def _custom_model_config(
    custom_data_loaders: type[LazyDataLoader] | list[type[LazyDataLoader]] = None,
    custom_anomaly_detectors: type[BaseDetector] | list[type[BaseDetector]] = None,
    custom_metrics: type[Metric] | list[type[Metric]] = None,
    custom_visualizers: (
        type[CustomDetectorVisualizer] | list[type[CustomDetectorVisualizer]]
    ) = None,
) -> dict[str, list[str]]:

    def _is_valid(cls: type) -> bool:
        # In case the model is defined in __main__
        if cls.__module__ == "__main__":
            warnings.warn(
                "Including a custom model in the demonstrator which is defined in the "
                "same file from which the demonstrator is started leads to run-time issues."
                "Please define these models in a separate .py file and import them into "
                f"your main script. The model {cls.__qualname__} will be ignored."
            )
            return False
        else:
            return True

    def _format(cls: type) -> str:
        return f"{cls.__module__}.{cls.__qualname__}"

    return {
        "data_loaders": [
            _format(data_loader)
            for data_loader in convert_to_list(custom_data_loaders or [])
            if _is_valid(data_loader)
        ],
        "anomaly_detectors": [
            _format(anomaly_detector)
            for anomaly_detector in convert_to_list(custom_anomaly_detectors or [])
            if _is_valid(anomaly_detector)
        ],
        "metrics": [
            _format(metric)
            for metric in convert_to_list(custom_metrics or [])
            if _is_valid(metric)
        ],
        "custom_visualizers": [
            _format(visualizer)
            for visualizer in convert_to_list(custom_visualizers or [])
            if _is_valid(visualizer)
        ],
    }
