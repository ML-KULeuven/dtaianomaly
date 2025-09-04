import streamlit as st

from dtaianomaly.anomaly_detection import BaseDetector, BaseNeuralDetector
from dtaianomaly.in_time_ad.CustomDetectorVisualizer import CustomDetectorVisualizer


class NeuralNetVisualizer(CustomDetectorVisualizer):

    def __init__(self):
        super().__init__(name="Show the network architecture", icon="🔗")

    def is_compatible(self, detector_type: type[BaseDetector]) -> bool:
        return issubclass(detector_type, BaseNeuralDetector)

    def show_custom_visualization(self, detector: BaseDetector) -> None:
        # Show the data
        st.markdown(
            "Below, you can see a simple overview of the network architecture, to get an "
            "idea of the different layers and how they are connected to each other."
        )
        st.write(detector.neural_network_)
