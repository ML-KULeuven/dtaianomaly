from typing import List

import streamlit as st

from dtaianomaly.anomaly_detection import Supervision
from dtaianomaly.data import DataSet
from dtaianomaly.in_time_ad._st_AnomalyDetector import StAnomalyDetector
from dtaianomaly.in_time_ad._utils import error_no_detectors
from dtaianomaly.in_time_ad._visualization import (
    plot_anomaly_scores,
    plot_detected_anomalies,
)
from dtaianomaly.preprocessing import MinMaxScaler
from dtaianomaly.thresholding import FixedCutoffThreshold


class StQualitativeEvaluator:

    @staticmethod
    def plot_anomaly_scores(
        data_set: DataSet, st_anomaly_detectors: List[StAnomalyDetector]
    ) -> List[str]:
        st.markdown(
            """
            Below figure shows the time series again, as well as the predicted anomaly scores. The
            higher the anomaly score, the more likely the observation is anomalous. A high-performing
            anomaly detector should predict large anomaly scores for the true anomalies!
            """
        )

        # Retrieve the decision scores
        decision_functions = StQualitativeEvaluator._get_decision_functions(
            st_anomaly_detectors, "plot_anomaly_scores"
        )
        if decision_functions is None:
            return []

        # Plot the decision functions
        st.plotly_chart(
            plot_anomaly_scores(
                X=data_set.X_test,
                y=data_set.y_test,
                feature_names=data_set.feature_names,
                time_steps=data_set.time_steps_test,
                anomaly_scores=decision_functions,
            )
        )

        return [
            "from dtaianomaly.visualization import plot_anomaly_scores",
            f"plot_anomaly_scores({StQualitativeEvaluator._get_used_data(data_set)}, y_pred)",
        ]

    @staticmethod
    def plot_detected_anomalies(
        data_set: DataSet, st_anomaly_detectors: List[StAnomalyDetector]
    ) -> List[str]:

        st.markdown(
            """
            To see which observations are identified as anomalies, we must convert the continuous
            anomaly scores into binary events: anomaly or not? Various strategies for this exist,
            of which arguably the easiest one is setting a threshold. First select one of the anomaly
            detectors on the left, then select a threshold, and see which points are detected as an
            anomaly!

            Instead of just showing which observations are flagged as an anomaly, we divide the
            observations into three categories:
            - **True Positives (TP):** the predicted anomalies that are actually anomalies.
            - **False Positives (FP):** the predicted anomalies that are no real anomalies but normal!
            - **False Negatives (FN):** the real anomalies that were not detected!
            """
        )

        # A container for the ordering
        container = st.container()

        # Retrieve the decision scores
        decision_functions = StQualitativeEvaluator._get_decision_functions(
            st_anomaly_detectors, "plot_detected_anomalies"
        )
        if decision_functions is None:
            return []

        with container:

            # Configure the anomaly detector
            col_detector, col_threshold = st.columns(2)
            selected_detector = col_detector.selectbox(
                label="Anomaly detector",
                options=decision_functions,
                label_visibility="collapsed",
            )

            # Configure the cutoff
            min_value = decision_functions[selected_detector].min()
            max_value = decision_functions[selected_detector].max()
            cutoff = col_threshold.slider(
                label="Threshold",
                min_value=min_value,
                max_value=max_value,
                value=0.9 * (max_value - min_value) + min_value,
                step=0.01,
                label_visibility="collapsed",
            )

        # Compute the binary decisions
        y_pred = FixedCutoffThreshold(cutoff=cutoff).threshold(
            decision_functions[selected_detector]
        )

        # Plot the detected anomalies
        st.plotly_chart(
            plot_detected_anomalies(
                X=data_set.X_test,
                y=data_set.y_test,
                y_pred=y_pred,
                feature_names=data_set.feature_names,
                time_steps=data_set.time_steps_test,
            )
        )

        # Load the data arrays
        compatible_supervision = data_set.compatible_supervision()
        if Supervision.SEMI_SUPERVISED in compatible_supervision:
            ground_truth = "y_test"
        else:
            ground_truth = "y"

        return [
            "from dtaianomaly.thresholding import FixedCutoff",
            "from dtaianomaly.visualization import plot_time_series_anomalies",
            f"y_pred_bin = FixedCutoff(cutoff={cutoff}).threshold(y_pred)",
            f"plot_time_series_anomalies({StQualitativeEvaluator._get_used_data(data_set)}, {ground_truth}, y_pred_bin)",
        ]

    @staticmethod
    def _get_used_data(data_set: DataSet) -> str:
        if Supervision.SEMI_SUPERVISED in data_set.compatible_supervision():
            return "X_test, y_test"
        else:
            return "X, y"

    @staticmethod
    def _get_decision_functions(
        st_anomaly_detectors: List[StAnomalyDetector], pills_key: str
    ):

        # Retrieve the raw anomaly scores
        decision_functions = {
            str(anomaly_detector.detector): anomaly_detector.decision_function_
            for anomaly_detector in st_anomaly_detectors
            if hasattr(anomaly_detector, "decision_function_")
            and anomaly_detector.decision_function_ is not None
        }
        if len(decision_functions) == 0:
            error_no_detectors()
            return

        # Normalize the anomaly scores
        with st.expander("Scaling anomaly scores", icon="🛠️"):
            st.markdown(
                """
                The meaning of the predicted anomaly scores depend on the used
                anomaly detectors. For example, the anomaly scores of distance-based
                methods like Matrix Profile correspond to distances, while the
                anomaly scores of Isolation Forest is related to the depth of the
                the trees. Because of this, it might often be difficult to compare
                the raw anomaly scores of multiple detectors directly. To cope
                with this, we offer methods to scale the anomaly scores and make
                them comparable. The following approaches are available:

                - **Raw decision scores.** Do not apply any scaling and show the
                  raw predicted anomaly scores.
                - **Min-max scaled.** Scale the anomaly scores linearly such that
                  they fall in the interval [0, 1].
                """
            )
            normalization_technique = st.pills(
                label="Anomaly scores normalization",
                options=["Raw decision scores", "Min-max scaled"],
                default="Min-max scaled",
                label_visibility="collapsed",
                key=pills_key,
            )
        if normalization_technique == "Min-max scaled":
            decision_functions = {
                name: MinMaxScaler().fit_transform(decision_function)[0]
                for name, decision_function in decision_functions.items()
            }

        return decision_functions
