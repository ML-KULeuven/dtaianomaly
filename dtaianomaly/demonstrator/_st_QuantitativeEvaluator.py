import copy

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dtaianomaly.anomaly_detection import Supervision
from dtaianomaly.data import DataSet
from dtaianomaly.demonstrator._st_AnomalyDetector import StAnomalyDetector
from dtaianomaly.demonstrator._utils import (
    get_parameters,
    input_widget_hyperparameter,
    show_class_summary,
    show_small_header,
    update_object,
)
from dtaianomaly.evaluation import Metric, ProbaMetric, ThresholdMetric
from dtaianomaly.preprocessing import MinMaxScaler
from dtaianomaly.thresholding import FixedCutoff


class StMetric:
    counter: int
    metric: Metric
    parameters: dict
    thresholding: FixedCutoff | None

    def __init__(self, counter: int, metric: type[Metric], configuration: dict):
        self.counter = counter

        # Initialize the parameters
        parameters, required_parameters = get_parameters(metric)
        self.parameters = {
            key: value
            for key, value in configuration["optional"].items()
            if key in parameters
        }

        # Initialize the metric
        self.metric = metric(
            **{
                key: value
                for key, value in configuration["required"].items()
                if key in required_parameters
            }
        )

        # Initialize the thresholding if no proba metric is given
        if not issubclass(metric, ProbaMetric):
            self.thresholding = FixedCutoff(configuration["default_threshold"])
            self.parameters["cutoff"] = configuration["optional"]["cutoff"]
        else:
            self.thresholding = None

    def show_metric(self) -> (bool, bool, "StMetric"):

        old_metric = copy.deepcopy(self)

        # Save some space for the header
        header = st.container()

        # Show an explanation of the detector
        show_class_summary(self.metric)

        # Select the hyperparameters, and update the detector if necessary
        col_select_parameters, col_update_parameters, col_remove = st.columns(3)
        with col_select_parameters.popover(
            label="Configure", icon=":material/settings:", use_container_width=True
        ):
            parameters = self.select_parameters()

        # Update the metric if requested
        do_update = col_update_parameters.button(
            label="Update parameters",
            key=f"update-metric-parameters-{self.counter}",
            use_container_width=True,
        )
        if do_update:
            thresholding_parameters = {}
            if "cutoff" in parameters:
                thresholding_parameters["cutoff"] = parameters.pop("cutoff")
            do_update = update_object(self.metric, parameters) or update_object(
                self.thresholding, thresholding_parameters
            )

        # Add a button to remove the detecor
        remove_metric = col_remove.button(
            label="Remove metric",
            icon="❌",
            key=f"remove_metric_{self.counter}",
            use_container_width=True,
        )

        with header:
            show_small_header(self.to_str())

        return do_update, remove_metric, old_metric

    def select_parameters(self) -> dict[str, any]:
        selected_parameters = {}

        # Add the other parameters
        for parameter, config in self.parameters.items():

            # Format the kwargs for the widget
            input_widget_kwargs = {
                key: value for key, value in config.items() if key != "type"
            }
            input_widget_kwargs["key"] = "-".join(
                [parameter, str(self.counter), config["type"], "metric"]
            )
            if "label" not in input_widget_kwargs:
                input_widget_kwargs["label"] = parameter

            selected_parameters[parameter] = input_widget_hyperparameter(
                config["type"], **input_widget_kwargs
            )

        return selected_parameters

    def compute_score(self, y_true: np.array, y_pred: np.array) -> float:
        y_pred = MinMaxScaler().fit_transform(y_pred)[0]
        if self.thresholding is not None:
            return ThresholdMetric(
                thresholder=self.thresholding, metric=self.metric
            ).compute(y_true, y_pred)
        else:
            return self.metric.compute(y_true, y_pred)

    def to_str(self):
        if self.thresholding is None:
            return str(self.metric)
        else:
            return f"{self.metric} [cutoff={self.thresholding.cutoff}]"

    def get_code_lines(self, data_set: DataSet) -> list[str]:
        # Load the data arrays
        compatible_supervision = data_set.compatible_supervision()
        if Supervision.SEMI_SUPERVISED in compatible_supervision:
            ground_truth = "y_test"
        else:
            ground_truth = "y"

        # Optional import of thresholding
        if isinstance(self.thresholding, FixedCutoff):
            return [
                f"from dtaianomaly.thresholding import {self.thresholding.__class__.__name__}",
                f"from dtaianomaly.evaluation import {self.metric.__class__.__name__}, ThresholdMetric",
                f"metric = ThresholdMetric(",
                f"  thresholder={self.thresholding}",
                f"  metric={self.metric}",
                f")",
                "score = metric.compute(y_test, y_pred)",
            ]
        else:
            return [
                f"from dtaianomaly.evaluation import {self.metric.__class__.__name__}",
                f"metric = {self.metric}",
                f"score = metric.compute({ground_truth}, y_pred)",
            ]


class StQualitativeEvaluationLoader:

    counter: int = 0
    default_metrics: list[type[Metric]]
    all_metrics: list[(str, type[Metric])]
    configuration: dict

    def __init__(self, all_metrics: list[(str, type[Metric])], configuration: dict):
        self.all_metrics = []
        self.default_metrics = []
        for name, cls in all_metrics:
            if name not in configuration["to-remove"]:
                self.all_metrics.append((name, cls))
            if name in configuration["default"] or name == configuration["default"]:
                self.default_metrics.append(cls)

        self.configuration = configuration

    def select_metric(self) -> StMetric | None:
        col_selection, col_button = st.columns([3, 1])
        selected_metric = col_selection.selectbox(
            label="Select anomaly detector",
            options=self.all_metrics,
            index=None,
            format_func=lambda t: t[0],
            label_visibility="collapsed",
        )
        button_clicked = col_button.button(
            label="Load metric", use_container_width=True
        )

        if button_clicked and selected_metric is not None:
            return self._load_metric(selected_metric[1])

        return None

    def select_default_metrics(self) -> list[StMetric]:
        return [self._load_metric(metric) for metric in self.default_metrics]

    def _load_metric(self, metric: type[Metric]) -> StMetric:
        st_metric = StMetric(
            counter=self.counter,
            metric=metric,
            configuration=self.configuration["parameters"],
        )

        # Update the counter and return the metric
        self.counter += 1
        return st_metric


class StEvaluationScores:
    scores: pd.DataFrame

    def __init__(
        self,
        detectors: list[StAnomalyDetector],
        metrics: list[StMetric],
        y_test: np.array,
    ):
        self.scores = pd.DataFrame(
            index=[str(detector.detector) for detector in detectors],
            columns=[metric.to_str() for metric in metrics],
        )
        for detector in detectors:
            for metric in metrics:
                self.add(detector, metric, y_test)

    def show_scores(self) -> None:

        # Show the scores in a bar-plot
        df_melted = self.scores.T.melt(
            ignore_index=False, var_name="Metric", value_name="value"
        )
        df_melted["x"] = df_melted.index
        fig = px.bar(df_melted, x="x", y="value", color="Metric", barmode="group")
        fig.update_layout(
            height=300, xaxis_title=None, yaxis_title=None, legend_title_text=None
        )
        st.plotly_chart(fig)

        # Show the raw scores
        st.dataframe(self.scores)

    def add(
        self, detector: StAnomalyDetector, metric: StMetric, y_test: np.array
    ) -> None:
        self.scores.loc[str(detector.detector), metric.to_str()] = metric.compute_score(
            y_test, detector.decision_function_
        )

    def remove_detector(self, detector: StAnomalyDetector) -> None:
        self.scores = self.scores.drop(index=str(detector.detector))

    def remove_metric(self, metric: StMetric) -> None:
        self.scores = self.scores.drop(columns=metric.to_str())
