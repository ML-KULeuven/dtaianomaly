import abc

from dtaianomaly.anomaly_detection import BaseDetector

__all__ = ["CustomDetectorVisualizer"]


class CustomDetectorVisualizer(abc.ABC):
    """
    Base class for custom detector visualizations.

    A base class for showing custom visualizations for anomaly detectors
    within InTimeAD.

    Parameters
    ----------
    name : str
        The name to use for this visualizer.
    icon : str, default=None
        The icon to show along the visualization. If None, then no icon will
        be shown.
    """

    name: str
    icon: str | None

    def __init__(self, name: str, icon: str | None):
        self.name = name
        self.icon = icon

    @abc.abstractmethod
    def is_compatible(self, detector: type[BaseDetector]) -> bool:
        """
        Check compatibility of the given detector.

        Check whether the given detector is compatible with this visualizer.

        Parameters
        ----------
        detector : BaseDetector-object
            The type of the anomaly detector to check if it is compatible.

        Returns
        -------
        bool
            True if and only if this visualizer is compatible with the given
            detector, and thus the visualization could be made for the detector.
        """

    @abc.abstractmethod
    def show_custom_visualization(self, detector: BaseDetector) -> None:
        """
        Show the custom visualization for the given anomaly detector.

        Show the additional information of the given anomaly detector that
        is useful for understanding the model.

        Parameters
        ----------
        detector : BaseDetector
            The anomaly detector for which the visualization should be made.
        """
