"""
This module contains functions for plotting time series. It can be imported as follows:

>>> from dtaianomaly import visualization

The functions within this module offer alternative manners to nicely plot the time series
along with the ground truth or predicted anomalies.
"""

from .visualization import (
    plot_time_series_colored_by_score,
    plot_time_series_anomalies,
    plot_demarcated_anomalies,
    plot_with_zoom,
    plot_anomaly_scores,
    format_time_steps,
)

__all__ = [
    "plot_time_series_colored_by_score",
    "plot_time_series_anomalies",
    "plot_demarcated_anomalies",
    "plot_with_zoom",
    "plot_anomaly_scores",
    "format_time_steps",
]
