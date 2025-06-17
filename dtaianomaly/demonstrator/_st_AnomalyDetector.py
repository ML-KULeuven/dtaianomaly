import numpy as np
import streamlit as st

from dtaianomaly.anomaly_detection import BaseDetector
from dtaianomaly.data import DataSet
from dtaianomaly.demonstrator._utils import (
    get_class_summary,
    get_parameters,
    write_code_lines,
)


class StAnomalyDetector:

    counter: int
    detector: BaseDetector
    parameters: dict
    decision_function_: np.array

    def __init__(self, counter: int, detector: type[BaseDetector], configuration: dict):
        self.counter = counter

        parameters, required_parameters = get_parameters(detector)
        self.parameters = {
            key: value
            for key, value in configuration["optional"].items()
            if key in parameters
        }
        self.detector = detector(
            **{
                key: value
                for key, value in configuration["required"].items()
                if key in required_parameters
            }
        )

    def show_anomaly_detector(self) -> bool:
        # TODO maybe add a button to actually update the parameters (and refit), instead of continuously updating them?
        #   Then also check if the detector actually changed, to avoid wasting resources
        col_remove, col_detector = st.columns([1, 6], vertical_alignment="center")

        with col_detector.expander(f"{self.detector}"):
            st.write(
                get_class_summary(self.detector)
            )  # TODO this could maybe be in some kind of nice box or highlight?
            self.show_hyperparameters()
            write_code_lines(self.get_code_lines(), use_expander=False)

        remove_detector = col_remove.button(
            label="Remove",
            icon="❌",
            key=f"remove_detector_{self.counter}",
            use_container_width=True,
        )

        return remove_detector

    def show_hyperparameters(self):
        # Put the window size first
        if "window_size" in self.parameters:
            window_sizes = {
                "Manual": "manual",
                "Dominant Fourier Frequency": "fft",
                "Highest Autocorrelation": "acf",
                "Summary Statistics Subsequence": "suss",
                "Multi-Window-Finder": "mwf",
            }
            index = 0
            for i, window_size in enumerate(window_sizes.values()):
                if self.detector.window_size == window_size:
                    index = i
            col1, col2 = st.columns(2)
            option = col1.selectbox(
                label="Window size",
                options=window_sizes,
                index=index,
                key=f"window_size_select_{self.counter}",
            )
            value = col2.number_input(
                label="Manual window size",
                min_value=self.parameters["window_size"]["min_value"],
                step=self.parameters["window_size"]["step"],
                value=self.parameters["window_size"]["value"],
                disabled=option != "Manual",
                key=f"window_size_number_{self.counter}",
            )
            if option == "Manual":
                self.detector.window_size = value
            else:
                self.detector.window_size = window_sizes[option]

        # Add the other parameters
        for parameter, config in self.parameters.items():
            if parameter == "window_size":
                continue

            # TODO

    def fit_predict(self, data_set: DataSet):
        with st.spinner(f"Fitting {self.detector}"):
            self.detector.fit(
                data_set.X_test, data_set.y_test
            )  # TODO use train data if possible?
        with st.spinner(f"Detecting anomalies with {self.detector}"):
            self.decision_function_ = self.detector.decision_function(data_set.X_test)

    def get_code_lines(self) -> list[str]:
        return [  # TODO depends on the detector and the dataset
            f"from dtaianomaly.anomaly_detection import {self.detector.__class__.__name__}",
            f"detector = {self.detector}.fit(X_train)",  # TODO does not work if dataset is unsupervised (+ for supervised models add training labels y_train)
            "y_pred = detector.predict_proba(X_test)",
        ]


class StAnomalyDetectorLoader:

    counter: int = 0
    default_anomaly_detector: type[BaseDetector]
    all_anomaly_detectors: list[(str, type[BaseDetector])]
    configuration: dict

    def __init__(self, all_anomaly_detectors: list[(str, type)], configuration: dict):
        self.all_anomaly_detectors = []
        for name, cls in all_anomaly_detectors:
            if name not in configuration["to-remove"]:
                self.all_anomaly_detectors.append((name, cls))
            if name == configuration["default"]:
                self.default_anomaly_detector = cls

        self.configuration = configuration

    def select_anomaly_detector(self) -> StAnomalyDetector | None:
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

    def select_default_anomaly_detector(self) -> StAnomalyDetector:
        return self._load_detector(self.default_anomaly_detector)

    def _load_detector(
        self, anomaly_detector: type[BaseDetector]
    ) -> "StAnomalyDetector":
        st_anomaly_detector = StAnomalyDetector(
            self.counter, anomaly_detector, self.configuration["parameters"]
        )
        self.counter += 1
        return st_anomaly_detector
