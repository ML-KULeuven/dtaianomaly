from typing import List, Tuple, Type

import streamlit as st

from dtaianomaly.evaluation import Metric


class StQualitativeEvaluator:
    default_evaluation_metric: Metric
    all_evaluation_metrics: List[Tuple[str, type]]

    def __init__(
        self,
        all_evaluation_metrics: List[Tuple[str, type]],
        default_evaluation_metric: Type[Metric],
    ):
        self.default_evaluation_metric = default_evaluation_metric()  # TODO?
        self.all_evaluation_metrics = all_evaluation_metrics

    def show_evaluation_metrics(self):
        st.warning("WIP")
