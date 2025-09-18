import numpy as np

from dtaianomaly.data import DataSet, LazyDataLoader


class DemonstrationTimeSeriesLoader(LazyDataLoader):
    """
    A data loader object to load the demonstration time series.

    The demonstration time series loader is a simple time series
    which typically follows a sine-wave of length 1400 containing
    approximately 12 periods. The time series generally follows
    the typical sine-pattern (including some Gaussian noise), the
    valley around observation 950 is slightly deeper than the other
    valleys in the time series, leading to an anomaly.
    """

    def _load(self) -> DataSet:
        X, y = demonstration_time_series()
        return DataSet(X_test=X, y_test=y)


def demonstration_time_series() -> (np.ndarray, np.ndarray):
    """
    Generate a time series for demonstration purposes. This is a noisy
    sine wave with one valley that is deeper than the other ones.

    Returns
    -------
    x: np.ndarray of shape (nb_samples)
        The raw time series data
    y: np.ndarray of shape (nb_samples)
        The ground truth labels
    """
    np.random.seed(42)

    X = np.sin(np.linspace(0, 25 * np.pi, 1400) + 2)
    X += np.random.normal(0, 0.1, X.shape)
    y = np.zeros(shape=X.shape[0])

    X[920:965] -= 0.5
    y[920:965] = 1

    return X, y


def inject_anomalies(
    time_series: np.ndarray,
    nb_anomalies: int = 10,
    min_anomaly_magnitude: float = 1.0,
    max_anomaly_magnitude: float = 2.0,
) -> np.array:
    """
    Inject random anomalies in the given time series. This method will
    only inject point anomalies by adding a random offset to some random
    observations in the time series. Note that this method will adapt the
    given time series in place.

    Parameters
    ----------
    time_series: array_like of shape (n_samples, n_features)
        The time series to inject anomalies in
    nb_anomalies: int, default=10
        The number of anomalies to inject
    min_anomaly_magnitude: float, default=1.0
        The minimum added magnitude for anomalies
    max_anomaly_magnitude: float, default=2.0
        The maximum added magnitude for anomalies

    Returns
    -------
    anomaly_labels: array-like of shape (n_samples)
        Binary anomaly labels, with a one indicating that an anomaly has
        been injected in the time series.
    """
    anomaly_indices = np.random.choice(
        time_series.shape[0], nb_anomalies, replace=False
    )
    if len(time_series.shape) == 1:
        anomalies = np.random.uniform(
            min_anomaly_magnitude, max_anomaly_magnitude, size=nb_anomalies
        )
        anomalies *= np.random.randint(2, size=anomalies.shape) * 2 - 1
        time_series[anomaly_indices] += anomalies
    else:
        anomalies = np.random.uniform(
            min_anomaly_magnitude,
            max_anomaly_magnitude,
            size=(nb_anomalies, time_series.shape[1]),
        )
        anomalies *= np.random.randint(2, size=anomalies.shape) * 2 - 1
        time_series[anomaly_indices, :] += anomalies
    anomaly_labels = np.zeros(shape=time_series.shape[0])
    anomaly_labels[anomaly_indices] = 1
    return anomaly_labels


def make_sine_wave(
    nb_samples: int,
    amplitude: float = 1.0,
    frequency: float = 5.0,
    phase: float = 0.0,
    noise_level: float = 0.2,
    seed: int = None,
    **kwargs,
) -> (np.ndarray, np.ndarray):
    """
    Generate a random sine wave and inject anomalies into it.

    Parameters
    ----------
    nb_samples: int
        The length of the sine wave.
    amplitude: float, default=1.0
        The amplitude of the sine wave, the max absolute value of the sine wave.
    frequency: float, default=5.0
        The frequency of the sine wave, the number of oscillations
    phase: float, default=0.0
        The phase of the sine wave, where the oscillation starts.
    noise_level: float, default=0.2
        The amount of Gaussian noise to add to the time series
    seed: int, default=None
        The seed for generating a random sine wave. If no value is provided,
        then the sine wave will be random.
    **kwargs:
        Parameters to pass to the ``inject_anomalies`` method.

    Returns
    -------
    x: np.ndarray of shape (nb_samples)
        The raw time series data
    y: np.ndarray of shape (nb_samples)
        The ground truth labels
    """
    # Generate the time series
    np.random.seed(seed)
    t = np.arange(nb_samples) / nb_samples
    nice_sine_wave = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    noisy_sine_wave = nice_sine_wave + np.random.normal(0, noise_level, nb_samples)
    noisy_sine_wave = noisy_sine_wave.reshape(-1, 1)
    # Inject anomalies
    anomaly_labels = inject_anomalies(noisy_sine_wave, **kwargs)
    # Create a DataSet object
    return noisy_sine_wave, anomaly_labels
