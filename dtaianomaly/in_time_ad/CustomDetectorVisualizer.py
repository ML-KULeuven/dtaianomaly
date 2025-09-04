import abc

from dtaianomaly.anomaly_detection import BaseDetector


class CustomDetectorVisualizer(abc.ABC):

    name: str
    icon: str | None

    def __init__(self, name: str, icon: str | None):
        self.name = name
        self.icon = icon

    @abc.abstractmethod
    def is_compatible(self, detector: type[BaseDetector]) -> bool:
        """
        Check whether the given detector is compatible with this visualizer.

        Parameters
        ----------
        detector: BaseDetector-object
            The type of the anomaly detector to check if it is compatible.

        Returns
        -------
        is_compatible: bool
            True if and only if this visualizer is compatible with the given
            detector, and thus the visualization could be made for the detector.
        """

    @abc.abstractmethod
    def show_custom_visualization(self, detector: BaseDetector) -> None:
        """
        Show the custom visualization for the given anomaly detector.

        Parameters
        ----------
        detector: BaseDetector
            The anomaly detector for which the visualization should be made.
        """
