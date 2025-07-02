import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dtaianomaly.anomaly_detection import (
    BaseDetector,
    KMeansAnomalyDetector,
    KShapeAnomalyDetector,
)
from dtaianomaly.demonstrator.CustomDetectorVisualizer import CustomDetectorVisualizer


class CentroidVisualizer(CustomDetectorVisualizer):

    def __init__(self):
        super().__init__(name="Show the centroids", icon=":material/graph_3:")

    def is_compatible(self, detector_type: type[BaseDetector]) -> bool:
        return (
            detector_type == KMeansAnomalyDetector
            or detector_type == KShapeAnomalyDetector
        )

    def show_custom_visualization(self, detector: BaseDetector) -> None:

        centroids = self._get_centroids(detector)

        # Create subplots: one column per time series
        fig = make_subplots(
            rows=1,
            cols=len(centroids),
            shared_yaxes=True,
            shared_xaxes=True,
            subplot_titles=[f"Centroid {i + 1}" for i in range(len(centroids))],
        )

        # Add each time series to its own column
        for i, centroid in enumerate(centroids):
            if len(centroid.shape) > 1:
                for j in range(centroid.shape[1]):
                    fig.add_trace(go.Scatter(y=centroid[:, j]), row=1, col=i + 1)
            else:
                fig.add_trace(go.Scatter(y=centroid), row=1, col=i + 1)

        # Layout options
        fig.update_layout(
            height=300,
            xaxis_title="Time",
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=False,
        )

        # Show the data
        st.markdown(
            "Below, you can see the centroids of the different clusters. These "
            "represent the different normal behaviors. If a subsequence has a "
            "large distance to all these subsequences, then it is different from"
            "the normal behaviors, and consequently an anomaly."
        )
        st.plotly_chart(fig)

    @staticmethod
    def _get_centroids(detector: BaseDetector) -> list[np.array]:
        if isinstance(detector, KShapeAnomalyDetector):
            return detector.centroids_
        elif isinstance(detector, KMeansAnomalyDetector):
            return [
                detector.k_means_.cluster_centers_[i, :].reshape(
                    detector.window_size_, -1
                )
                for i in range(detector.k_means_.cluster_centers_.shape[0])
            ]
