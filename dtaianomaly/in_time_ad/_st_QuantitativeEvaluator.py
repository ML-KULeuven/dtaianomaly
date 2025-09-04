import copy
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dtaianomaly.anomaly_detection import Supervision
from dtaianomaly.data import DataSet
from dtaianomaly.evaluation import Metric, ProbaMetric, ThresholdMetric
from dtaianomaly.in_time_ad._st_AnomalyDetector import StAnomalyDetector
from dtaianomaly.in_time_ad._utils import (
    get_parameters,
    input_widget_hyperparameter,
    show_class_summary,
    show_small_header,
    update_object,
)
from dtaianomaly.in_time_ad._visualization import get_detector_color_map
from dtaianomaly.preprocessing import MinMaxScaler
from dtaianomaly.thresholding import FixedCutoff


class StMetric:
    __METRIC_COUNTER: int = 0
    metric_id: int
    metric: Metric
    parameters: dict
    thresholding: FixedCutoff | None

    def __init__(self, metric: type[Metric], configuration: dict):
        self.metric_id = StMetric.__METRIC_COUNTER
        StMetric.__METRIC_COUNTER += 1

        # Initialize the parameters
        parameters, required_parameters = get_parameters(metric)
        self.parameters = {
            key: value
            for key, value in configuration["parameters-optional"].items()
            if key in parameters
        }

        # Initialize the metric
        self.metric = metric(
            **{
                key: value
                for key, value in configuration["parameters-required"].items()
                if key in required_parameters
            }
        )

        # Initialize the thresholding if no proba metric is given
        if not issubclass(metric, ProbaMetric):
            self.thresholding = FixedCutoff(
                configuration["parameters-required"]["cutoff"]
            )
            self.parameters["cutoff"] = configuration["parameters-optional"]["cutoff"]
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
            key=f"update-metric-parameters-{self.metric_id}",
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
            key=f"remove_metric_{self.metric_id}",
            use_container_width=True,
        )

        with header:
            show_small_header(str(self))

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
                [parameter, str(self.metric_id), config["type"], "metric"]
            )
            if "label" not in input_widget_kwargs:
                input_widget_kwargs["label"] = parameter

            selected_parameters[parameter] = input_widget_hyperparameter(
                config["type"], **input_widget_kwargs
            )

        return selected_parameters

    def compute_score(self, y_true: np.array, y_pred: np.array) -> float:
        if y_pred is None:  # If no scores are available
            return np.nan

        y_pred = MinMaxScaler().fit_transform(y_pred)[0]
        if self.thresholding is not None:
            return ThresholdMetric(
                thresholder=self.thresholding, metric=self.metric
            ).compute(y_true, y_pred)
        else:
            return self.metric.compute(y_true, y_pred)

    def __str__(self):
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

        if self.metric.__module__.startswith("dtaianomaly.evaluation."):
            imports = [
                f"from dtaianomaly.evaluation import {self.metric.__class__.__name__}, ThresholdMetric"
            ]
        else:
            imports = [
                f"from dtaianomaly.evaluation import ThresholdMetric",
                f"from {self.metric.__module__} import {self.metric.__class__.__name__}",
            ]

        # Optional import of thresholding
        if isinstance(self.thresholding, FixedCutoff):
            return (
                [
                    f"from dtaianomaly.thresholding import {self.thresholding.__class__.__name__}",
                ]
                + imports
                + [
                    f"metric = ThresholdMetric(",
                    f"  thresholder={self.thresholding}",
                    f"  metric={self.metric}",
                    f")",
                    "score = metric.compute(y_test, y_pred)",
                ]
            )
        else:

            module = self.metric.__module__
            if module.startswith("dtaianomaly.evaluation."):
                module = "dtaianomaly.evaluation"

            return [
                f"from {module}  import {self.metric.__class__.__name__}",
                f"metric = {self.metric}",
                f"score = metric.compute({ground_truth}, y_pred)",
            ]


class StQualitativeEvaluationLoader:

    default_metrics: list[type[Metric]]
    all_metrics: list[(str, type[Metric])]
    configuration: dict

    def __init__(self, all_metrics: list[(str, type[Metric])], configuration: dict):
        self.all_metrics = []
        self.default_metrics = []
        for name, cls in all_metrics:
            if name not in configuration["exclude"]:
                self.all_metrics.append((name, cls))
            if name in configuration["default"] or name == configuration["default"]:
                self.default_metrics.append(cls)
        self.all_metrics = sorted(self.all_metrics, key=lambda x: x[0])

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

        if selected_metric is not None:
            show_class_summary(selected_metric[1])

        button_clicked = col_button.button(
            label="Load metric", use_container_width=True
        )

        if button_clicked and selected_metric is not None:
            return self._load_metric(selected_metric[1])

        return None

    def select_default_metrics(self) -> list[StMetric]:
        return [self._load_metric(metric) for metric in self.default_metrics]

    def _load_metric(self, metric: type[Metric]) -> StMetric:
        return StMetric(
            metric=metric,
            configuration=self.configuration,
        )


class StEvaluationScores:
    detectors: dict[int, str]
    metrics: dict[int, str]
    scores: pd.DataFrame

    def __init__(
        self,
        detectors: list[StAnomalyDetector],
        metrics: list[StMetric],
        y_test: np.array,
    ):
        self.detectors = {}
        self.metrics = {}
        self.scores = pd.DataFrame(
            index=[detector.detector_id for detector in detectors],
            columns=[metric.metric_id for metric in metrics],
        )
        for detector in detectors:
            for metric in metrics:
                self.add(detector, metric, y_test)

    def show_scores(self) -> None:

        # Identify duplicated metrics and decide which columns to drop
        metric_ids = defaultdict(list)
        for metric_id, metric_name in self.metrics.items():
            metric_ids[metric_name].append(metric_id)
        to_drop = []
        for metric, ids in metric_ids.items():
            if len(ids) > 1:
                to_drop.extend(ids[1:])
                st.warning(
                    f"Metric '{metric}' is defined {len(ids)} times. The evaluation will only be shown once."
                )
        formatted_scores = self.scores.drop(columns=to_drop).rename(
            columns=self.metrics, index=self.detectors
        )

        # Define a color map
        color_map = get_detector_color_map(formatted_scores.index)

        # Show the scores in a bar-plot
        df_melted = formatted_scores.T.melt(
            ignore_index=False, var_name="Metric", value_name="value"
        )
        df_melted["x"] = df_melted.index

        fig = px.bar(
            df_melted,
            x="x",
            y="value",
            color="Metric",
            barmode="group",
            color_discrete_map=color_map,
        )
        fig.update_layout(
            height=300, xaxis_title=None, yaxis_title=None, legend_title_text=None
        )
        st.plotly_chart(fig)

        # Show the raw scores
        st.dataframe(formatted_scores)

        # Download the
        st.download_button(
            label="Download the scores as a csv-file",
            data=formatted_scores.to_csv().encode("utf-8"),
            file_name="scores.csv",
            mime="text/csv",
            icon=":material/download:",
        )

    def add(
        self, detector: StAnomalyDetector, metric: StMetric, y_test: np.array
    ) -> None:
        self.scores.loc[detector.detector_id, metric.metric_id] = metric.compute_score(
            y_test, detector.decision_function_
        )
        self.detectors[detector.detector_id] = str(detector)
        self.metrics[metric.metric_id] = str(metric)

    def remove_detector(self, detector: StAnomalyDetector) -> None:
        self.scores = self.scores.drop(index=detector.detector_id)
        self.detectors.pop(detector.detector_id)

    def remove_metric(self, metric: StMetric) -> None:
        self.scores = self.scores.drop(columns=metric.metric_id)
        self.metrics.pop(metric.metric_id)
