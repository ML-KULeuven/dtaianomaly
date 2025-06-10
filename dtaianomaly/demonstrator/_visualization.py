from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dtaianomaly import utils
from dtaianomaly.evaluation._common import make_intervals


def plot_data(
    X: np.ndarray,
    y: Optional[np.array],
    feature_names: Optional[List[str]],
    time_steps: Optional[np.array],
    fig: go.Figure = None,
    row: int = 1,
    col: int = 1,
) -> go.Figure:

    if fig is None:
        fig = make_subplots(rows=1, cols=1)

    # Format the time steps
    if time_steps is None:
        time_steps = np.arange(X.shape[0])

    # Format the feature names
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(utils.get_dimension(X))]

    # Plot the data
    if utils.is_univariate(X):
        fig.add_trace(
            go.Scatter(x=time_steps, y=X, mode="lines", name=feature_names[0]),
            row=row,
            col=col,
        )
    else:
        for d in range(utils.get_dimension(X)):
            fig.add_trace(
                go.Scatter(
                    x=time_steps, y=X[:, d], mode="lines", name=feature_names[d]
                ),
                row=row,
                col=col,
            )

    # Plot the labels
    if y is not None:
        starts, ends = make_intervals(y)
        for s, e in zip(starts, ends):
            fig.add_vrect(
                x0=s,
                x1=e,
                line_width=3,
                line_color="red",
                fillcolor="red",
                opacity=0.2,
                row=row,
                col=col,
            )

    # Format the figure
    fig.update_layout(
        height=300,
        xaxis_title="Time",
        margin=dict(l=0, r=0, t=0, b=0),
    )

    # Return the figure
    return fig


def plot_anomaly_scores(
    X: np.ndarray,
    y: Optional[np.array],
    feature_names: Optional[List[str]],
    time_steps: Optional[np.array],
    anomaly_scores: Dict[str, np.array],
) -> go.Figure:

    # Initialize the figure
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

    # Plot the data
    plot_data(
        X=X,
        y=y,
        feature_names=feature_names,
        time_steps=time_steps,
        fig=fig,
        row=1,
        col=1,
    )

    # Plot the anomaly scores
    for name, score in anomaly_scores.items():
        fig.add_trace(
            go.Scatter(x=time_steps, y=score, mode="lines", name=name), row=2, col=1
        )

    return fig


def plot_detected_anomalies(
    X: np.ndarray,
    y: np.array,
    y_pred: np.array,
    feature_names: Optional[List[str]],
    time_steps: Optional[np.array],
):

    # Format the time steps
    if time_steps is None:
        time_steps = np.arange(X.shape[0])

    # Plot the data already
    fig = plot_data(
        X=X,
        y=None,  # No need to mark the ground truth anomaly directly
        feature_names=feature_names,
        time_steps=time_steps,
    )

    # Handle both multivariate an univariate time series (creat new variable to avoid modifying the array)
    if utils.is_univariate(X):
        X_ = np.reshape(X, shape=(X.shape[0], 1))
    else:
        X_ = X

    for d in range(utils.get_dimension(X)):
        # Plot the true positives
        true_positive = (y == 1) & (y_pred == 1)
        fig.add_trace(
            go.Scatter(
                x=time_steps[true_positive],
                y=X_[true_positive, d],
                mode="markers",
                name=f"TP ({true_positive.sum()})",
                marker={"color": "green"},
                showlegend=(d == 0),
            )
        )

        # Plot the false positives
        false_positive = (y == 0) & (y_pred == 1)
        fig.add_trace(
            go.Scatter(
                x=time_steps[false_positive],
                y=X_[false_positive, d],
                mode="markers",
                name=f"FP ({false_positive.sum()})",
                marker={"color": "red"},
                showlegend=(d == 0),
            )
        )

        # Plot the false negatives
        false_negative = (y == 1) & (y_pred == 0)
        fig.add_trace(
            go.Scatter(
                x=time_steps[false_negative],
                y=X_[false_negative, d],
                mode="markers",
                name=f"FN ({false_negative.sum()})",
                marker={"color": "orange"},
                showlegend=(d == 0),
            )
        )

    return fig
