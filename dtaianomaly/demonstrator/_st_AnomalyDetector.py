from typing import List, Optional, Tuple, Type

import numpy as np
import streamlit as st

from dtaianomaly.anomaly_detection import BaseDetector
from dtaianomaly.data import DataSet


class StAnomalyDetectorLoader:

    counter: int = 0
    default_anomaly_detector: Type[BaseDetector]
    all_anomaly_detectors: List[Tuple[str, type]]

    def __init__(
        self,
        all_anomaly_detectors: List[Tuple[str, type]],
        default_anomaly_detector: Type[BaseDetector],
    ):
        self.default_anomaly_detector = default_anomaly_detector
        self.all_anomaly_detectors = all_anomaly_detectors

    def select_anomaly_detector(self) -> Optional["StAnomalyDetector"]:
        col_selection, col_button = st.columns([3, 1])
        selected_detector = col_selection.selectbox(
            label="Select anomaly detector",
            options=self.all_anomaly_detectors,
            index=None,
            format_func=lambda t: t[0],
            label_visibility="collapsed",
        )
        button_clicked = col_button.button(
            label="Load detector", use_container_width=True
        )
        if button_clicked and selected_detector is not None:
            return self._load_detector(selected_detector[1])

    def select_default_anomaly_detector(self) -> "StAnomalyDetector":
        return self._load_detector(self.default_anomaly_detector)

    def _load_detector(
        self, anomaly_detector: Type[BaseDetector]
    ) -> "StAnomalyDetector":
        st_anomaly_detector = StAnomalyDetector(self.counter, anomaly_detector)
        self.counter += 1
        return st_anomaly_detector


class StAnomalyDetector:

    counter: int
    detector: BaseDetector
    decision_function_: np.array

    def __init__(self, counter: int, detector: Type[BaseDetector]):
        self.counter = counter
        self.detector = detector(window_size=100)  # TODO fix

    def show_anomaly_detector(self) -> bool:

        col_detector, col_run, col_remove, col_hyperparameters = st.columns(
            [3, 1, 1, 1]
        )

        col_detector.write(self.detector)
        # col_run.button('Detect anomalies', key=f'run_detector_{self.counter}')
        remove_detector = col_remove.button(
            "Remove", key=f"remove_detector_{self.counter}"
        )
        col_hyperparameters.popover(
            "Hyperparameters", icon=":material/settings:", use_container_width=True
        )

        # TODO add functionality -> to configure the parameters

        return remove_detector

    def fit_predict(self, data_set: DataSet):
        with st.spinner(f"Fitting {self.detector}"):
            self.detector.fit(
                data_set.X_test, data_set.y_test
            )  # TODO use train data if possible?
        with st.spinner(f"Detecting anomalies with {self.detector}"):
            self.decision_function_ = self.detector.decision_function(data_set.X_test)
