import tempfile
from typing import List, Tuple

import streamlit as st

from dtaianomaly.anomaly_detection import Supervision
from dtaianomaly.data import DataSet, LazyDataLoader, PathDataLoader
from dtaianomaly.demonstrator._visualization import plot_data


class StDataLoader:

    data_set: DataSet
    initial_index: int
    data_loader: LazyDataLoader
    all_data_loaders: List[Tuple[str, type]]

    def __init__(
        self,
        all_data_loaders: List[Tuple[str, type]],
        configuration: dict,
    ):
        self.all_data_loaders = []
        for i, (name, cls) in enumerate(all_data_loaders):
            if name not in configuration["to-remove"]:
                self.all_data_loaders.append((name, cls))
            if name == configuration["default"]:
                self.initial_index = i
                self.data_loader = cls()
                self.data_set = self.data_loader.load()

        self.configuration = configuration

    def select_data_loader(self) -> bool:
        col_selection, col_configuration, col_button = st.columns([1, 0.5, 0.5])

        # Select a data loader
        _, data_loader_cls = col_selection.selectbox(
            label="Select an anomaly detector",
            options=self.all_data_loaders,
            index=self.initial_index,
            format_func=lambda t: t[0],
            label_visibility="collapsed",
        )

        # Configure the data loader
        # At this point, we only care about a loaders that use a file and no other parameters (as other ones don't exist as of now)
        parameters = {}
        with col_configuration.popover(
            "Configuration", icon=":material/settings:", use_container_width=True
        ):
            if issubclass(data_loader_cls, PathDataLoader):
                uploaded_file = st.file_uploader("Upload a file")
                if uploaded_file is not None:
                    file_name = uploaded_file.name
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=file_name
                    ) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        parameters["path"] = tmp_file.name

        # A button to actually load the data
        button_clicked = col_button.button(label="Load data", use_container_width=True)
        if button_clicked:
            try:
                self.data_loader = data_loader_cls(**parameters)
                self.data_set = self.data_loader.load()
                if (
                    "path" in parameters
                ):  # This is not recommended in practice, but for showing a nice file (without the temporary path)
                    self.data_loader.path = file_name

            except Exception as e:
                st.exception(e)
                st.error(
                    "Something went wrong, did you correctly configure the data loader?",
                    icon="🚨",
                )

        # Return whether the data was updated
        return button_clicked

    def show_data(self):
        figs = {"train": None, "test": None}

        # Create the figure for the train data
        if self.data_set.X_train is not None:
            figs["train"] = plot_data(
                X=self.data_set.X_train,
                y=self.data_set.y_train,
                feature_names=self.data_set.feature_names,
                time_steps=self.data_set.time_steps_train,
            )

        # Create figure for the train data
        figs["test"] = plot_data(
            X=self.data_set.X_test,
            y=self.data_set.y_test,
            feature_names=self.data_set.feature_names,
            time_steps=self.data_set.time_steps_test,
        )

        # Add the figures to the streamlit-page
        if self.data_set.X_train is not None:
            tabs = st.tabs(["Train data", "Test data"])
            tabs[0].plotly_chart(figs["train"], key="loaded-data-train")
            tabs[1].plotly_chart(figs["test"], key="loaded-data-test")
        else:
            st.plotly_chart(figs["test"], key="loaded-data-test")

    def get_code_lines(self) -> List[str]:
        code_lines = [
            f"from dtaianomaly.data import {self.data_loader.__class__.__name__}",
            f"data_loader = {self.data_loader}",
            "data_set = data_loader.load()",
        ]

        # Add the train data, depending on the data
        compatible_supervision = self.data_set.compatible_supervision()
        if Supervision.SUPERVISED in compatible_supervision:
            code_lines += ["X_train, y_train = data_set.X_train, data_set.y_train"]
        elif Supervision.SEMI_SUPERVISED in compatible_supervision:
            code_lines += ["X_train = data_set.X_train"]

        # Add the test data
        code_lines += ["X_test, y_test = data_set.X_test, data_set.y_test"]

        # Return the code lines
        return code_lines
