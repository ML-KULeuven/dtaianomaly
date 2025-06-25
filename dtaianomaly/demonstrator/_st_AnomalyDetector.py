import copy

import numpy as np
import streamlit as st

from dtaianomaly.anomaly_detection import (
    BaseDetector,
    Supervision,
    check_is_valid_window_size,
)
from dtaianomaly.data import DataSet
from dtaianomaly.demonstrator._utils import (
    get_parameters,
    input_widget_hyperparameter,
    show_class_summary,
    show_small_header,
    update_object,
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
            for key, value in configuration["parameters-optional"].items()
            if key in parameters
        }
        self.detector = detector(
            **{
                key: value
                for key, value in configuration["parameters-required"].items()
                if key in required_parameters
            }
        )
        if "window_size" in self.parameters:
            self.parameters["window_size_selection"] = configuration[
                "parameters-optional"
            ]["window_size_selection"]

    def show_anomaly_detector(self) -> (bool, bool, "StAnomalyDetector"):
        old_detector = copy.deepcopy(self)

        # Save some space for the header
        header = st.container()

        # Show an explanation of the detector
        show_class_summary(self.detector)

        # Select the hyperparameters, and update the detector if necessary
        col_select_hyperparameters, col_update_hyperparameters, remove_col = st.columns(
            3
        )
        with col_select_hyperparameters.popover(
            label="Configure", icon=":material/settings:", use_container_width=True
        ):
            hyperparameters = self.select_hyperparameters()

        # Update the model if requested
        do_update = col_update_hyperparameters.button(
            label="Update hyperparameters",
            key=f"update-detector-hyperparameters-{self.counter}",
            use_container_width=True,
        )
        if do_update:
            do_update = update_object(self.detector, hyperparameters)

        # Add a button to remove the detecor
        remove_detector = remove_col.button(
            label="Remove detector",
            icon="❌",
            key=f"remove_detector_{self.counter}",
            use_container_width=True,
        )

        with header:
            show_small_header(self.detector)

        return do_update, remove_detector, old_detector

    def select_hyperparameters(self) -> dict[str, any]:

        # A dictionary for the selected hyperparameters
        selected_hyperparameters = {}

        # Add the other parameters
        for parameter, config in self.parameters.items():
            if parameter == "window_size_selection":
                continue

            # Format the kwargs for the widget
            input_widget_kwargs = {
                key: value for key, value in config.items() if key != "type"
            }
            input_widget_kwargs["key"] = "-".join(
                [parameter, str(self.counter), config["type"], "detector"]
            )
            if "label" not in input_widget_kwargs:
                input_widget_kwargs["label"] = parameter

            # For the window size, we add some additional logic
            if parameter == "window_size":
                # Select the way in which the window size is computed
                col_select, col_value = st.columns(2)
                _, selected_window_size = col_select.selectbox(
                    format_func=lambda t: t[0],
                    key=f"window_size_select_{self.counter}",
                    **self.parameters["window_size_selection"],
                )
                try:
                    check_is_valid_window_size(selected_window_size)
                    valid_window_size = True
                except:
                    valid_window_size = False

                input_widget_kwargs["disabled"] = valid_window_size

                # Select the manual window size
                with col_value:
                    selected_window_size_integer = input_widget_hyperparameter(
                        config["type"], **input_widget_kwargs
                    )

                # Add the value to the dictionary of selected hyperparameters
                if not valid_window_size:
                    selected_hyperparameters[parameter] = selected_window_size_integer
                else:
                    selected_hyperparameters[parameter] = selected_window_size

            else:
                selected_hyperparameters[parameter] = input_widget_hyperparameter(
                    config["type"], **input_widget_kwargs
                )

        return selected_hyperparameters

    def fit_predict(self, data_set: DataSet):
        # Check for compatibility
        if not data_set.is_compatible(self.detector):
            st.warning(
                f"Anomaly detector {self.detector} is not compatible with the data!"
            )
            self.decision_function_ = np.zeros_like(data_set.y_test)
            return

        # Retrieve the correct data
        if Supervision.SEMI_SUPERVISED in data_set.compatible_supervision():
            X_train, y_train = data_set.X_train, data_set.y_train
        else:
            X_train, y_train = data_set.X_test, data_set.y_test

        # Fit the detector
        with st.spinner(f"Fitting {self.detector}"):
            self.detector.fit(X_train, y_train)

        # Compute the decision scores
        with st.spinner(f"Detecting anomalies with {self.detector}"):
            self.decision_function_ = self.detector.decision_function(data_set.X_test)

    def get_code_lines(self, data_set: DataSet) -> list[str]:

        compatible_supervision = data_set.compatible_supervision()
        if Supervision.SUPERVISED in compatible_supervision:
            train_data = "X_train, y_train"
            test_data = "X_test"
        elif Supervision.SEMI_SUPERVISED in compatible_supervision:
            train_data = "X_train"
            test_data = "X_test"
        else:
            train_data = "X"
            test_data = "X"

        module = self.detector.__module__
        if module.startswith("dtaianomaly.anomaly_detection."):
            module = "dtaianomaly.anomaly_detection"

        return [
            f"from {module} import {self.detector.__class__.__name__}",
            f"detector = {self.detector}.fit({train_data})",
            f"y_pred = detector.predict_proba({test_data})",
        ]


class StAnomalyDetectorLoader:

    counter: int = 0
    default_detectors: list[type[BaseDetector]]
    all_anomaly_detectors: list[(str, type[BaseDetector])]
    configuration: dict

    def __init__(
        self,
        all_anomaly_detectors: list[(str, type[BaseDetector])],
        configuration: dict,
    ):
        self.all_anomaly_detectors = []
        self.default_detectors = []
        for name, cls in all_anomaly_detectors:
            if name not in configuration["exclude"]:
                self.all_anomaly_detectors.append((name, cls))
            if name in configuration["default"] or name == configuration["default"]:
                self.default_detectors.append(cls)

        self.all_anomaly_detectors = sorted(
            self.all_anomaly_detectors, key=lambda x: x[0]
        )
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

        if selected_detector is not None:
            show_class_summary(selected_detector[1])

        button_clicked = col_button.button(
            label="Load detector", use_container_width=True
        )
        if button_clicked and selected_detector is not None:
            return self._load_detector(selected_detector[1])

    def select_default_anomaly_detector(self) -> list[StAnomalyDetector]:
        return [self._load_detector(detector) for detector in self.default_detectors]

    def _load_detector(
        self, anomaly_detector: type[BaseDetector]
    ) -> "StAnomalyDetector":
        st_anomaly_detector = StAnomalyDetector(
            counter=self.counter,
            detector=anomaly_detector,
            configuration=self.configuration,
        )
        self.counter += 1
        return st_anomaly_detector
