import matplotlib.pyplot as plt
import numpy as np

def plot_time_series_colored_by_score(X: np.ndarray, y: np.ndarray, ax: plt.Axes = None, nb_colors: int = 100, **kwargs) -> plt.Figure:
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
    y_min, y_max = y.min(), y.max()
    y_scaled = (y - y_min) / (y_max - y_min) if y_max > y_min else np.zeros_like(y)
    y_binned = [np.floor(score * nb_colors) / nb_colors for score in y_scaled]
    colormap = plt.get_cmap('RdYlGn', nb_colors).reversed()
    for i in range(0, X.shape[0]-1):
        color = colormap(y_binned[i])
        ax.plot([i, i+1], X[[i, i+1]], c=color)
    return plt.gcf()

def plot_time_series_anomalies(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, ax: plt.Axes = None, **kwargs) -> plt.Figure:
    """
    Visualizes time series data with true and predicted anomalies, highlighting true positives (TP),
    false positives (FP), and false negatives (FN).

    Parameters
    ----------
    X: np.ndarray of shape (n_samples, n_attributes)
        The time series to plot
    y_true: np.ndarray of shape (n_samples,)
        Ground truth anomaly labels (binary values: 0 or 1).
    y_pred: np.ndarray of shape (n_samples,)
        Predicted anomaly labels (binary values: 0 or 1).
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

    # Check if all predicted values are binary.
    if not np.all(np.isin(y_pred, [0, 1])):
        raise ValueError('The predicted anomaly scores must be binary.')

    # Identify TP, FP, FN
    TP = (y_true == 1) & (y_pred == 1)
    FP = (y_true == 0) & (y_pred == 1)
    FN = (y_true == 1) & (y_pred == 0)

    # Plot the time series
    ax.plot(np.arange(len(X)), X, label='Time Series', color='blue', alpha=0.5)

    # Scatter points for TP, FP, FN
    ax.scatter(np.arange(len(X))[TP], X[TP], color='green', label='TP')
    ax.scatter(np.arange(len(X))[FP], X[FP], color='red', label='FP')
    ax.scatter(np.arange(len(X))[FN], X[FN], color='orange', label='FN')

    # Customize the plot
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Real Values', fontsize=12)
    ax.set_title('Time Series Anomaly Detection', fontsize=15)
    ax.legend()
    ax.grid()

    return plt.gcf()


