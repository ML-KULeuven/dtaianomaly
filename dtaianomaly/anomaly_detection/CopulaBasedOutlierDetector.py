from pyod.models.copod import COPOD

from dtaianomaly.anomaly_detection.BaseDetector import Supervision
from dtaianomaly.anomaly_detection.PyODAnomalyDetector import PyODAnomalyDetector


class CopulaBasedOutlierDetector(PyODAnomalyDetector):
    """
    Copula-based outlier detector (COPD) algorithm.

    COPOD [li2020copod]_ is based on modeling multivariate data distributions
    using Copula models. Copula functions separate the marginal distributions
    from the dependency structure of a multivariate distribution. This allows
    a copula to describe the joint distribution over the features using only
    the independent marginals, offering high flexibility when modeling high
    dimensional data. Outliers are consequently detected by measuring the
    tail probabilities. COPOD is parameter-free because the copula function
    does not involve learning or stochastic training.

    Notes
    -----
    The COPOD detector inherets from :py:class:`~dtaianomaly.anomaly_detection.PyODAnomalyDetector`.

    Parameters
    ----------
    window_size: int or str
        The window size to use for extracting sliding windows from the time series. This
        value will be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.
    stride: int, default=1
        The stride, i.e., the step size for extracting sliding windows from the time series.
    **kwargs
        Arguments to be passed to the PyOD COPOD detector.

    Attributes
    ----------
    window_size_: int
        The effectively used window size for this anomaly detector
    pyod_detector_ : COPOD
        A COPOD detector of PyOD

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import CopulaBasedOutlierDetector
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> copod = CopulaBasedOutlierDetector(10).fit(x)
    >>> copod.decision_function(x)
    array([ 9.90110663,  9.67868282,  9.51525285, ..., 25.00182389,
           24.60594424, 24.30393026])

    References
    ----------
    .. [li2020copod] Li, Z., Zhao, Y., Botta, N., Ionescu, C. and Hu, X.
       COPOD: Copula-Based Outlier Detection. IEEE International Conference
       on Data Mining (ICDM), 2020.
    """

    def _initialize_detector(self, **kwargs) -> COPOD:
        return COPOD(**kwargs)

    def _supervision(self) -> Supervision:
        return Supervision.UNSUPERVISED
