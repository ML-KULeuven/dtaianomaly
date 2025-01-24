from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch

from dtaianomaly import utils


def plot_time_series_colored_by_score(
    X: np.ndarray,
    y: np.ndarray,
    time_steps: np.array = None,
    ax: plt.Axes = None,
    nb_colors: int = 100,
    **kwargs,
) -> plt.Figure:
    """
    Plots the given time series, and color it according to the given scores.
    Higher scores will be colored red, and lower scores will be colored green.
    Thus, if the ground truth anomaly scores are passed, red corresponds to
    anomalies and green to normal observations.

    Parameters
    ----------
    X: np.ndarray of shape (n_samples, n_attributes)
        The time series to plot
    y: np.ndarray of shape (n_samples)
        The scores, according to which the plotted data should be colored.
    time_steps: np.array of shape (n_samples), default=None
        The time steps to plot. If no time steps are provided, then the
        default range ``[0, ..., n_samples-1]`` will be used.
    ax: plt.Axes, default=None
        The axes onto which the plot should be made. If None, then a new
        figure and axis will be created.
    nb_colors: int, default=100
        The number of colors to use for plotting the time series.
    **kwargs:
        Arguments to be passed to plt.Figure(), in case ``ax=None``.

    Returns
    -------
    fig: plt.Figure
        The figure containing the plotted data.

    Notes
    -----
    Each segment in the time series will be plotted independently. Thus,
    for time series with many observations, plotting the data using this
    method can cost a huge amount of time.
    """
    if ax is None:
        plt.figure(**kwargs)
        ax = plt.gca()

    # Format the time steps
    time_steps = format_time_steps(time_steps, X.shape[0])

    y_min, y_max = y.min(), y.max()
    y_scaled = (y - y_min) / (y_max - y_min) if y_max > y_min else np.zeros_like(y)
    y_binned = [np.floor(score * nb_colors) / nb_colors for score in y_scaled]
    colormap = plt.get_cmap("RdYlGn", nb_colors).reversed()
    for i in range(0, X.shape[0] - 1):
        color = colormap(y_binned[i])
        ax.plot([time_steps[i], time_steps[i + 1]], X[[i, i + 1]], c=color)
    return plt.gcf()


def plot_time_series_anomalies(
    X: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    time_steps: np.array = None,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Figure:
    """
    Visualizes time series data with true and predicted anomalies, highlighting true positives (TP),
    false positives (FP), and false negatives (FN).

    Parameters
    ----------
    X: np.ndarray of shape (n_samples, n_attributes)
        The time series to plot
    y: np.ndarray of shape (n_samples,)
        Ground truth anomaly labels (binary values: 0 or 1).
    y_pred: np.ndarray of shape (n_samples,)
        Predicted anomaly labels (binary values: 0 or 1).
    time_steps: np.array of shape (n_samples), default=None
        The time steps to plot. If no time steps are provided, then the
        default range ``[0, ..., n_samples-1]`` will be used.
    ax: plt.Axes, default=None
        The axes onto which the plot should be made. If None, then a new
        figure and axis will be created.
    **kwargs:
        Arguments to be passed to plt.Figure(), in case ``ax=None``.

    Returns
    -------
    fig: plt.Figure
        The figure containing the plotted data.
    """

    # Prepare the axis
    if ax is None:
        plt.figure(**kwargs)
        ax = plt.gca()

    # Check if the given y values are binary
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("The predicted anomaly scores must be binary.")
    if not np.all(np.isin(y_pred, [0, 1])):
        raise ValueError("The predicted anomaly scores must be binary.")

    # Identify TP, FP, FN
    TP = (y == 1) & (y_pred == 1)
    FP = (y == 0) & (y_pred == 1)
    FN = (y == 1) & (y_pred == 0)

    # Format the time steps
    time_steps = format_time_steps(time_steps, X.shape[0])

    # Plot the time series
    ax.plot(time_steps, X)

    # Scatter points for TP, FP, FN
    X_reshaped = X.reshape((-1, utils.get_dimension(X)))
    tps, fps, fns = None, None, None
    for i in range(utils.get_dimension(X)):
        tps = ax.scatter(time_steps[TP], X_reshaped[TP, i], color="green")
        fps = ax.scatter(time_steps[FP], X_reshaped[FP, i], color="red")
        fns = ax.scatter(time_steps[FN], X_reshaped[FN, i], color="orange")
    ax.legend([tps, fps, fns], ["TP", "FP", "FN"])

    return plt.gcf()


def plot_demarcated_anomalies(
    X: np.ndarray,
    y: np.array,
    ax: plt.Axes = None,
    time_steps: np.array = None,
    color_anomaly: str = "red",
    alpha_anomaly: float = 0.2,
    **kwargs,
) -> plt.Figure:
    """
    Plot the given time series and binary anomaly labels. Each anomalous
    interval is marked by a colored area, depending on the provided parameters.

    Parameters
    ----------
    X: np.ndarray of shape (n_samples, n_attributes)
        The time series to plot
    y: np.array of shape (n_samples)
        The binary anomaly scores.
    ax: plt.Axes, default=None
        The axes onto which the plot should be made. If None, then a new
        figure and axis will be created.
    time_steps: np.array of shape (n_samples), default=None
        The time steps to plot. If no time steps are provided, then the
        default range ``[0, ..., n_samples-1]`` will be used.
    color_anomaly: str, default='red'
        The color in which the anomaly should be marked.
    alpha_anomaly: float, default=0.2
        The alpha value for marking the anomaly, to adjust transparency.
    **kwargs:
        Arguments to be passed to plt.Figure(), in case ``ax=None``.

    Returns
    -------
    fig: plt.Figure
        The figure containing the plotted data.
    """
    # Check if y is binary
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("The predicted anomaly scores must be binary!")

    # Initialize an axis object if none has been given
    if ax is None:
        plt.figure(**kwargs)
        ax = plt.gca()

    # Identify the anomalous regions
    diff = np.diff(y, prepend=0, append=0)
    start_events = np.where(diff == 1)[0]
    end_events = np.where(diff == -1)[0]

    # Format the time steps
    time_steps = format_time_steps(time_steps, X.shape[0])

    # Plot the time series data
    ax.plot(time_steps, X)

    # Plot the anomalous zones
    for start, end in zip(start_events, end_events):
        ax.axvspan(
            time_steps[start],
            time_steps[min(end, time_steps.shape[0] - 1)],
            color=color_anomaly,
            alpha=alpha_anomaly,
        )

    # Return the active figure
    return plt.gcf()


def plot_with_zoom(
    X: np.ndarray,
    y: np.array,
    start_zoom: int,
    end_zoom: int,
    time_steps: np.array = None,
    y_pred: np.array = None,
    method_to_plot=plot_demarcated_anomalies,
    color: str = "blue",
    linewidth: float = 3,
    linestyle: str = "--",
    **kwargs,
) -> plt.Figure:
    """
    Plot the given data in two axes, one showing the entire time
    series and one zooming in on a specific area of the time series.

    Parameters
    ----------
    X: np.ndarray of shape (n_samples, n_attributes)
        The time series to plot
    y: np.array of shape (n_samples)
        The binary anomaly scores.
    start_zoom: int
        The index in the data at which the zoom starts.
    end_zoom: int
        The index in the data at which the zoom ends.
    time_steps: np.array of shape (n_samples), default=None
        The time steps to plot. If no time steps are provided, then the
        default range ``[0, ..., n_samples-1]`` will be used.
    y_pred: np.array of shape (n_samples), default=None
        The predicted anomaly scores to plot. Is necessary if the
        ``method_to_plot`` requires predicted anomaly scores.
    method_to_plot: callable, default=:py:autofunc:`~dtaianomaly.visualization.plot_demarcated_anomalies`
        Method used for plotting the data. Should take as inputs
        the values ``X`` (the time series data), ``y`` (the anomaly
        labels``), ``time_steps`` (the time steps at which there was
        an observation) and ``ax`` (the axis on which the plot should
        be made). Optionally, the method takes as input a value ``y_pred``
        for the predicted anomaly scores.
    color: string, default='blue'
        The color of the lines to demarcate the area of zooming.
    linewidth: float, default=3
        The width of the lines to demarcate the area of zooming.
    linestyle: str, default='--'
        The style of the lines to demarcate the area of zooming.
    **kwargs:
        Arguments to be passed to plt.subplots().

    Returns
    -------
    fig: plt.Figure
        The figure containing the plotted data.
    """
    # Create the main figure and two subplots (axes)
    fig, (ax_main, ax_zoom) = plt.subplots(2, 1, **kwargs)

    # Format the time steps
    time_steps = format_time_steps(time_steps, X.shape[0])

    # Plot the data
    X_zoom = X[start_zoom:end_zoom]
    y_zoom = y[start_zoom:end_zoom]
    time_stamps_zoom = time_steps[start_zoom:end_zoom]
    if y_pred is None:
        method_to_plot(X=X, y=y, ax=ax_main, time_steps=time_steps)
        method_to_plot(X=X_zoom, y=y_zoom, ax=ax_zoom, time_steps=time_stamps_zoom)
    else:
        method_to_plot(X=X, y=y, y_pred=y_pred, ax=ax_main, time_steps=time_steps)
        method_to_plot(
            X=X_zoom,
            y=y_zoom,
            y_pred=y_pred[start_zoom:end_zoom],
            ax=ax_zoom,
            time_steps=time_stamps_zoom,
        )

    # Draw vertical lines to demarcate the area in which is zoomed
    for ax in [ax_main, ax_zoom]:
        for x in [start_zoom, end_zoom]:
            ax.axvline(
                x=time_steps[x], color=color, linestyle=linestyle, linewidth=linewidth
            )

    # Connect the demarcations across the subplots
    fig.add_artist(
        ConnectionPatch(
            xyA=(time_steps[start_zoom], ax_main.get_ylim()[0]),
            coordsA=ax_main.transData,
            xyB=(time_steps[start_zoom], ax_zoom.get_ylim()[1]),
            coordsB=ax_zoom.transData,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )
    )
    fig.add_artist(
        ConnectionPatch(
            xyA=(time_steps[end_zoom], ax_main.get_ylim()[0]),
            coordsA=ax_main.transData,
            xyB=(time_steps[end_zoom], ax_zoom.get_ylim()[1]),
            coordsB=ax_zoom.transData,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )
    )

    return fig


def plot_anomaly_scores(
    X: np.array,
    y: np.array,
    y_pred: np.array,
    time_steps: np.array = None,
    method_to_plot=plot_demarcated_anomalies,
    confidence: np.array = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot the given data with the ground truth anomalies, and compare the
    predicted anomaly scores.

    Parameters
    ----------
    X: np.ndarray of shape (n_samples, n_attributes)
        The time series to plot
    y: np.array of shape (n_samples)
        The binary anomaly scores.
    y_pred: np.array of shape (n_samples)
        The predicted anomaly scores to plot.
    time_steps: np.array of shape (n_samples), default=None
        The time steps to plot. If no time steps are provided, then the
        default range ``[0, ..., n_samples-1]`` will be used.
    method_to_plot: callable, default=:py:autofunc:`~dtaianomaly.visualization.plot_demarcated_anomalies`
        Method used for plotting the data along with the ground truth
        anomaly scores. Should take as inputs the values ``X`` (the
        time series data), ``y`` (the anomaly labels``), ``time_steps``
        (the time steps at which there was an observation) and ``ax``
        (the axis on which the plot should be made).
    confidence: np.array of shape (n_samples), default=None
        The confidence of the anomaly scores.
    **kwargs:
        Arguments to be passed to plt.subplots().

    Returns
    -------
    fig: plt.Figure
        The figure containing the plotted data.
    """
    # Create the figure
    fig, (ax_data, ax_pred) = plt.subplots(nrows=2, ncols=1, sharex=True, **kwargs)

    # Format the time steps
    time_steps = format_time_steps(time_steps, X.shape[0])

    # Plot the time series data
    ax_data.set_title("Time series data")
    method_to_plot(X=X, y=y, ax=ax_data, time_steps=time_steps)

    # Plot the anomaly scores
    ax_pred.set_title("Predicted anomaly scores")
    ax_pred.plot(time_steps, y_pred, label="Anomaly scores")

    # Predict the confidence interval
    if confidence is not None:
        ax_pred.fill_between(
            time_steps,
            y_pred - (1 - confidence),
            y_pred + (1 - confidence),
            color="gray",
            alpha=0.5,
            label="Confidence range",
        )
        ax_pred.legend()

    # Return the figure
    return fig


def format_time_steps(time_steps: Optional[np.array], n_samples: int) -> np.array:
    """
    Format the given time steps, to ensure that fixed time steps are
    provided in case they are ``None``. These fixed time steps will
    equal the range ``[0, ..., n_samples-1]``

    Parameters
    ----------
    time_steps: np.array of shape (n_samples) or None
        The time seps to format.
    n_samples: int
        The number of samples for which there should be a time step.

    Returns
    -------
    formatted_time_steps: np.array of shape (n_samples)
        If the given ``time_steps`` did not equal ``None``, then these
        values are returned. Otherwise, an array with values ``[0, ...,
        n_samples-1]`` is returned.
    """
    return np.arange(n_samples) if time_steps is None else time_steps
