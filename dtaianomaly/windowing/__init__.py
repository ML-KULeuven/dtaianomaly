"""
A common parameter in many time series anomaly detection algorithms is the
window size. This module functionality to create a sliding window, but also
to automatically compute the window size. It can be imported as follows:

>>> from dtaianomaly import windowing

The available identifiers for automatically computing the window size for
a given time series can be accessed as follows:

>>> from dtaianomaly.windowing import AUTO_WINDOW_SIZE_COMPUTATION
>>> AUTO_WINDOW_SIZE_COMPUTATION
['fft', 'acf', 'mwf', 'suss']

A predefined type is declared for the valid window sizes which can be passed
to :py:func:`~dtaianomaly.windowing.compute_window_size`:

>>> from dtaianomaly.windowing import WINDOW_SIZE_TYPE
>>> WINDOW_SIZE_TYPE  # doctest: +SKIP
int | typing.Literal['fft', 'acf', 'mwf', 'suss']

.. note::

    The implementations in this module are based on the following repository:
    https://github.com/ermshaua/window-size-selection

"""

from typing import Literal

from ._acf import highest_autocorrelation
from ._compute_window_size import compute_window_size
from ._fft import dominant_fourier_frequency
from ._mwf import multi_window_finder
from ._reverse_sliding_window import reverse_sliding_window
from ._sliding_window import sliding_window
from ._suss import summary_statistics_subsequences

AUTO_WINDOW_SIZE_COMPUTATION = ["fft", "acf", "mwf", "suss"]
WINDOW_SIZE_TYPE = int | Literal["fft", "acf", "mwf", "suss"]


__all__ = [
    "highest_autocorrelation",
    "compute_window_size",
    "dominant_fourier_frequency",
    "multi_window_finder",
    "reverse_sliding_window",
    "sliding_window",
    "summary_statistics_subsequences",
    "AUTO_WINDOW_SIZE_COMPUTATION",
    "WINDOW_SIZE_TYPE",
]
