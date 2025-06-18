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

    def show_anomaly_detector(self) -> (bool, bool):

        # Save some space for the header
        header = st.container()

        # Show an explanation of the detector
        self.show_explanation()

        # Select the hyperparameters, and update the detector if necessary
        col_select_hyperparameters, col_update_hyperparameters, col3 = st.columns(3)
        with col_select_hyperparameters.popover(
            label="Configure", icon=":material/settings:", use_container_width=True
        ):
            hyperparameters = self.select_hyperparameters()
        updated_hyperparameters = self.filter_updated_hyperparameters(hyperparameters)
        do_update = (
            col_update_hyperparameters.button(
                label="Update hyperparameters",
                key=f"update-hyperparameters-{self.counter}",
                use_container_width=True,
            )
            and len(updated_hyperparameters) > 0
        )
        if do_update:
            self.set_hyperparameters(updated_hyperparameters)

        # Add a button to remove the detecor
        remove_detector = col3.button(
            label="Remove detector",
            icon="❌",
            key=f"remove_detector_{self.counter}",
            use_container_width=True,
        )

        # Update the header at the end, to make sure that the parameters are correctly formatted
        header.markdown(f"##### {self.detector}")

        return do_update, remove_detector

    def show_explanation(self) -> None:
        st.markdown(get_class_summary(self.detector))

    def select_hyperparameters(self) -> dict[str, any]:

        # Method for selecting the input widget
        def _input_widget_hyperparameter(
            **kwargs,
        ) -> any:  # TODO add the parameters to the config
            if config["type"] == "number_input":
                return st.number_input(**kwargs)
            elif config["type"] == "select_slider":
                return st.select_slider(**kwargs)
            elif config["type"] == "toggle":
                return st.toggle(**kwargs)
            elif config["type"] == "number_input":
                return st.checkbox(**kwargs)
            elif config["type"] == "number_input":
                return st.pills(**kwargs)
            elif config["type"] == "number_input":
                return st.segmented_control(**kwargs)
            elif config["type"] == "number_input":
                return st.selectbox(**kwargs)

        # A dictionary for the selected hyperparameters
        selected_hyperparameters = {}

        # Add the other parameters
        for parameter, config in self.parameters.items():

            # Format the kwargs for the widget
            input_widget_kwargs = {
                key: value for key, value in config.items() if key != "type"
            }
            input_widget_kwargs["key"] = "-".join(
                [parameter, str(self.counter), config["type"]]
            )
            if "label" not in input_widget_kwargs:
                input_widget_kwargs["label"] = parameter

            # For the window size, we add some additional logic
            if parameter == "window_size":
                window_size_options = [
                    (
                        "Manual",
                        "Manual",
                        "Manually set the window size to a specific size.",
                    ),
                    (
                        "Dominant Fourier Frequency",
                        "fft",
                        "Use the window size which corresponds to the dominant frequency in the Fourier domain.",
                    ),
                    (
                        "Highest Autocorrelation",
                        "acf",
                        "Use the window size which corresponds to maximum autocorrelation.",
                    ),
                    ("Summary Statistics Subsequence", "suss", "help suss"),  # TODO
                    ("Multi-Window-Finder", "mwf", "help mwf"),  # TODO
                ]
                # Select the way in which the window size is computed
                col_select, col_value = st.columns(2)
                _, selected_window_size, _ = col_select.selectbox(
                    label="Window size",
                    options=window_size_options,
                    format_func=lambda t: t[0],
                    key=f"window_size_select_{self.counter}",
                    help="\n".join(
                        ["The used method for setting the window size:"]
                        + [
                            f"- **{full_name}:** {help_description}"
                            for full_name, _, help_description in window_size_options
                        ]
                    ),
                )
                input_widget_kwargs["disabled"] = selected_window_size != "Manual"

                # Select the manual window size
                with col_value:
                    selected_window_size_integer = _input_widget_hyperparameter(
                        **input_widget_kwargs
                    )

                # Add the value to the dictionary of selected hyperparameters
                if selected_window_size == "Manual":
                    selected_hyperparameters[parameter] = selected_window_size_integer
                else:
                    selected_hyperparameters[parameter] = selected_window_size

            else:
                selected_hyperparameters[parameter] = _input_widget_hyperparameter(
                    **input_widget_kwargs
                )

        return selected_hyperparameters

    def filter_updated_hyperparameters(
        self, hyperparameters: dict[str, any]
    ) -> dict[str, any]:
        return {
            param: value
            for param, value in hyperparameters.items()
            if getattr(self.detector, param) != value
        }

    def set_hyperparameters(self, hyperparameters: dict[str, any]) -> None:
        for param, value in hyperparameters.items():
            setattr(self.detector, param, value)

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
