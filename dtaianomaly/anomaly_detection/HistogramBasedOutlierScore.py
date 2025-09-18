from typing import Literal

from pyod.models.hbos import HBOS

from dtaianomaly.anomaly_detection.BaseDetector import Supervision
from dtaianomaly.anomaly_detection.PyODAnomalyDetector import PyODAnomalyDetector


class HistogramBasedOutlierScore(PyODAnomalyDetector):
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
    alpha: float in [0, 1], default=0.1
        The regularizer for preventing overflow.
    tol: float in [0, 1], default=0.5
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

    Notes
    -----
    The HBOS detector inherets from :py:class:`~dtaianomaly.anomaly_detection.PyODAnomalyDetector`.
    """

    n_bins: int | Literal["auto"]
    alpha: float
    tol: float

    def __init__(
        self,
        window_size: int | str,
        stride: int = 1,
        n_bins: int | Literal["auto"] = 10,
        alpha: float = 0.1,
        tol: float = 0.5,
        **kwargs,
    ):

        if isinstance(n_bins, str):
            if n_bins != "auto":
                raise ValueError("If `n_bins` is a string, it should equal 'auto'.")
        else:
            if not isinstance(n_bins, int) or isinstance(n_bins, bool):
                raise TypeError("`n_bins` should be integer or 'auto'")
            if n_bins <= 1:
                raise ValueError("`n_bins` should be at least 2")

        if not isinstance(alpha, (float, int)) or isinstance(alpha, bool):
            raise TypeError("`alpha` should be numeric")
        if alpha <= 0 or alpha >= 1:
            raise ValueError("`alpha` must be in [0, 1]")

        if not isinstance(tol, (float, int)) or isinstance(tol, bool):
            raise TypeError("`tol` should be numeric")
        if tol <= 0 or tol >= 1:
            raise ValueError("`tol` must be in [0, 1]")

        self.n_bins = n_bins
        self.alpha = alpha
        self.tol = tol

        super().__init__(window_size, stride, **kwargs)

    def _initialize_detector(self, **kwargs) -> HBOS:
        return HBOS(n_bins=self.n_bins, alpha=self.alpha, tol=self.tol, **kwargs)

    def _supervision(self):
        return Supervision.UNSUPERVISED
