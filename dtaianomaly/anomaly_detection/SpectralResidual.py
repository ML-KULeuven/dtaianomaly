import numpy as np

from dtaianomaly import utils
from dtaianomaly.anomaly_detection.BaseDetector import BaseDetector, Supervision


class SpectralResidual(BaseDetector):
    """
    Detect anomalies using Spectral Residual :cite:`hansheng2019time`.

    Compute anomaly scores based on the spectral residuals. The time series
    if first transformed to the frequency domain by applying the Fourier
    transform. Then, the amplitudes and their logarithm are computed, after
    which the spectral residuals are computed based on the difference between
    the logarithmic amplitudes and a moving average of the logarithmic
    amplitudes. The obtained residuals are converted back to the time domain
    using the inverse Fourier transform, and the obtained saliency map is
    returned as anomaly scores.

    Parameters
    ----------
    moving_average_window_size: int
        The size of the window that is used for applying a moving average over the
        logarithmic amplitudes in the fourrier domain.
    epsilon: int, default=1e-8
        A threshold on the amplitude to avoid invalid values.

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import SpectralResidual
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> sr = SpectralResidual(10).fit(x)
    >>> sr.decision_function(x)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([0.28706631, 0.19033123, 0.15518059, ..., 0.17092832, 0.1530504 ,
           0.27486696]...)

    Notes
    -----
    SpectralResidual only handles univariate time series.
    """

    moving_average_window_size: int
    epsilon: float

    def __init__(self, moving_average_window_size: int, epsilon: float = 1e-8):
        super().__init__(Supervision.UNSUPERVISED)

        if not isinstance(moving_average_window_size, int) or isinstance(
            moving_average_window_size, bool
        ):
            raise TypeError("`moving_average_window_size` should be an integer")
        if moving_average_window_size <= 0:
            raise ValueError(
                f"Variable 'moving_average_window_size' must be at least 1, received '{moving_average_window_size}'!"
            )

        if not isinstance(epsilon, (float, int)) or isinstance(epsilon, bool):
            raise TypeError("`epsilon` should be an integer")
        if epsilon <= 0:
            raise ValueError(
                f"Variable 'epsilon' must be at least 0, received '{epsilon}'!"
            )

        self.moving_average_window_size = moving_average_window_size
        self.epsilon = epsilon

    def _fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> None:

        # Check if the given dataset is univariate
        if not utils.is_univariate(X):
            raise ValueError("Input must be univariate!")

    def _decision_function(self, X: np.ndarray) -> np.array:
        # Based on:
        # - https://github.com/microsoft/anomalydetector/blob/a3260ea0ddfb868986b924a245e003a97143f9df/msanomalydetector/spectral_residual.py#L118
        # - https://github.com/microsoft/anomalydetector/blob/a3260ea0ddfb868986b924a245e003a97143f9df/msanomalydetector/util.py#L51

        # Check if the given dataset is univariate
        if not utils.is_univariate(X):
            raise ValueError("Input must be univariate!")

        # Make sure the time series array has only one dimension
        X = X.squeeze()

        # Compute the Fourier transform
        fourier_transform = np.fft.fft(X)

        # Compute the amplitudes
        amplitude = np.sqrt(fourier_transform.real**2 + fourier_transform.imag**2)
        index_smaller_than_epsilon = np.where(amplitude <= self.epsilon)[0]
        amplitude[index_smaller_than_epsilon] = self.epsilon

        # Compute the logarithm of the amplitudes
        log_amplitude = np.log(amplitude)
        index_smaller_than_epsilon[index_smaller_than_epsilon] = 0

        # Compute the moving average
        n = min(self.moving_average_window_size, X.shape[0])
        moving_average = np.cumsum(log_amplitude, dtype=float)
        moving_average[n:] = moving_average[n:] - moving_average[:-n]
        moving_average[n:] = moving_average[n:] / n
        for i in range(1, n):
            moving_average[i] /= i + 1

        # Compute the spectral
        spectral = np.exp(log_amplitude - moving_average)

        # Update the real part of the fourier transform
        fourier_transform.real = fourier_transform.real * spectral / amplitude
        fourier_transform.real[index_smaller_than_epsilon] = 0

        # Update the imaginary part of the fourier transform
        fourier_transform.imag = fourier_transform.imag * spectral / amplitude
        fourier_transform.imag[index_smaller_than_epsilon] = 0

        # Apply inverse Fourier transform
        inverse_fourier = np.fft.ifft(fourier_transform)

        # Compute the saliency map in the time domain
        saliency_map = np.sqrt(inverse_fourier.real**2 + inverse_fourier.imag**2)

        return saliency_map


if __name__ == "__main__":
    import doctest

    doctest.testmod()
