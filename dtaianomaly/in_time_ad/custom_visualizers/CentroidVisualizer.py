import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dtaianomaly.anomaly_detection import (
    BaseDetector,
    ClusterBasedLocalOutlierFactor,
    KMeansAnomalyDetector,
    KShapeAnomalyDetector,
)
from dtaianomaly.in_time_ad.CustomDetectorVisualizer import CustomDetectorVisualizer


class CentroidVisualizer(CustomDetectorVisualizer):

    def __init__(self):
        super().__init__(name="Show the centroids", icon="📍")

    def is_compatible(self, detector_type: type[BaseDetector]) -> bool:
        return (
            detector_type == KMeansAnomalyDetector
            or detector_type == KShapeAnomalyDetector
            or detector_type == ClusterBasedLocalOutlierFactor
        )

    def show_custom_visualization(self, detector: BaseDetector) -> None:

        st.markdown(
            "Below, you can see the centroids of the different clusters. These "
            "represent the different normal behaviors. If a subsequence has a "
            "large distance to all these subsequences, then it is different from"
            "the normal behaviors, and consequently an anomaly."
        )

        centroids = self._get_centroids(detector)
        cols = st.columns([len(c) for c in centroids.values()])
        for col, (title, separate_centroids) in zip(cols, centroids.items()):

            # Create subplots: one column per time series
            fig = make_subplots(
                rows=1,
                cols=len(separate_centroids),
                shared_yaxes=True,
                shared_xaxes=True,
            )
            # Add each time series to its own column
            for i, centroid in enumerate(separate_centroids):
                if len(centroid.shape) > 1:
                    for j in range(centroid.shape[1]):
                        fig.add_trace(go.Scatter(y=centroid[:, j]), row=1, col=i + 1)
                else:
                    fig.add_trace(go.Scatter(y=centroid), row=1, col=i + 1)

            # Layout options
            fig.update_layout(
                title_text=title,
                height=150,
                xaxis_title="Time",
                margin=dict(l=0, r=0, t=20, b=0),
                showlegend=False,
            )

            # Show the data
            col.plotly_chart(fig)

    @staticmethod
    def _get_centroids(detector: BaseDetector) -> dict[str, list[np.array]]:
        if isinstance(detector, KShapeAnomalyDetector):
            return {"The centroids": detector.centroids_}
        elif isinstance(detector, KMeansAnomalyDetector):
            return {
                "The centroids": [
                    detector.k_means_.cluster_centers_[i, :].reshape(
                        detector.window_size_, -1
                    )
                    for i in range(detector.k_means_.cluster_centers_.shape[0])
                ]
            }
        elif isinstance(detector, ClusterBasedLocalOutlierFactor):
            return {
                "The centroids of large clusters": [
                    detector.pyod_detector_.cluster_centers_[i, :].reshape(
                        detector.window_size_, -1
                    )
                    for i in detector.pyod_detector_.large_cluster_labels_
                ],
                "The centroids of small clusters": [
                    detector.pyod_detector_.cluster_centers_[i, :].reshape(
                        detector.window_size_, -1
                    )
                    for i in detector.pyod_detector_.small_cluster_labels_
                ],
            }
