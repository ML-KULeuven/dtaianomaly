from typing import Literal

from pyod.models.hbos import HBOS

from dtaianomaly.anomaly_detection._BaseDetector import Supervision
from dtaianomaly.anomaly_detection._BasePyODAnomalyDetector import (
    BasePyODAnomalyDetector,
)
from dtaianomaly.type_validation import (
    FloatAttribute,
    IntegerAttribute,
    LiteralAttribute,
)
from dtaianomaly.windowing import WINDOW_SIZE_TYPE

__all__ = ["HistogramBasedOutlierScore"]


class HistogramBasedOutlierScore(BasePyODAnomalyDetector):
    """
    Anomaly detector based on the Histogram Based Outlier Score (HBOS) algorithm :cite:`goldstein2012histogram`.

    Histogram Based Outlier Score (HBOS)  constructs for each feature
    a univariate histogram. Bins with a small height (for static bin widths) or wider bins (for
    dynamic bin widths) correspond to sparse regions of the feature space. Thus, values falling
    in these bins lay in sparse regions of the feature space and are considered more anomalous.

    In this implementation, it is possible to set a window size to take the past observations into
    account. However, HBOS assumes feature independence. Therefore, for a time series with :math:`D`
    attributes and a window size :math:`w`, HBOS constructs :math:`D \\times w` independent histograms,
    from which the anomaly score is computed.

    Parameters
    ----------
    window_size: int or str
        The window size to use for extracting sliding windows from the time series. This
        value will be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.
    stride: int, default=1
        The stride, i.e., the step size for extracting sliding windows from the time series.
    n_bins: int or 'auto', default=10
        The number of bins for each feature. If ``'auto'``, the birge-rozenblac method is used
        for automatically selecting the number of bins for each feature.
    alpha: float in ]0, 1[, default=0.1
        The regularizer for preventing overflow.
    tol: float in ]0, 1[, default=0.5
        Parameter defining the flexibility for dealing with samples that fall
        outside the bins.
    **kwargs
        Arguments to be passed to the PyOD histogram based outlier score.

    Attributes
    ----------
    window_size_: int
        The effectively used window size for this anomaly detector
    pyod_detector_ : HBOS
        An HBOS detector of PyOD

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import HistogramBasedOutlierScore
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> hbos = HistogramBasedOutlierScore(1).fit(x)
    >>> hbos.decision_function(x)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([0.51808795, 0.51808795, 0.51808795, ..., 0.48347552, 0.48347552, 0.48347552]...)
    """

    n_bins: int | Literal["auto"]
    alpha: float
    tol: float

    attribute_validation = {
        "n_bins": IntegerAttribute(minimum=2) | LiteralAttribute("auto"),
        "alpha": FloatAttribute(
            0.0, 1.0, inclusive_minimum=False, inclusive_maximum=False
        ),
        "tol": FloatAttribute(
            0.0, 1.0, inclusive_minimum=False, inclusive_maximum=False
        ),
    }

    def __init__(
        self,
        window_size: WINDOW_SIZE_TYPE,
        stride: int = 1,
        n_bins: int | Literal["auto"] = 10,
        alpha: float = 0.1,
        tol: float = 0.5,
        **kwargs,
    ):
        self.n_bins = n_bins
        self.alpha = alpha
        self.tol = tol
        super().__init__(window_size, stride, **kwargs)

    def _initialize_detector(self, **kwargs) -> HBOS:
        return HBOS(n_bins=self.n_bins, alpha=self.alpha, tol=self.tol, **kwargs)

    def _supervision(self):
        return Supervision.UNSUPERVISED
