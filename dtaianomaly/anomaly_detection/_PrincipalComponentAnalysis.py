from pyod.models.pca import PCA

from dtaianomaly.anomaly_detection._BaseDetector import Supervision
from dtaianomaly.anomaly_detection._BasePyODAnomalyDetector import (
    BasePyODAnomalyDetector,
)
from dtaianomaly.type_validation import FloatAttribute, IntegerAttribute, NoneAttribute
from dtaianomaly.windowing import WINDOW_SIZE_TYPE

__all__ = ["PrincipalComponentAnalysis"]


class PrincipalComponentAnalysis(BasePyODAnomalyDetector):
    """
    Anomaly detector based on the Principal Component Analysis (PCA) :cite:`aggarwal2017linear`.

    PCA maps the data to a lower dimensional space
    through linear projections. The goal of these projections is to
    capture the most important information of the samples. This important
    information is related to the type of behaviors that occur frequently
    in the data. Thus, anomalies are detected by measuring the deviation
    of the samples in the lower dimensional space.

    Parameters
    ----------
    window_size : int or str
        The window size to use for extracting sliding windows from the time series. This
        value will be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.
    stride : int, default=1
        The stride, i.e., the step size for extracting sliding windows from the time series.
    n_components : int or float, default=None
        The number of components to keep.

        - if ``int``: Use the specified number of components.
        - if ``float``: Use the percentage of components.
        - if ``None``: Use all components.

    **kwargs
        Arguments to be passed to the PyOD PCA.

    Attributes
    ----------
    window_size_ : int
        The effectively used window size for this anomaly detector
    pyod_detector_ : PCA
        A PCA-detector of PyOD

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import PrincipalComponentAnalysis
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> pca = PrincipalComponentAnalysis(10).fit(x)
    >>> pca.decision_function(x)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([16286.63724327, 15951.05917741, 15613.5739773 , ..., 18596.5273311 , 18496.96613747, 18483.47985554]...)
    """

    n_components: int | float | None

    attribute_validation = {
        "n_components": IntegerAttribute(1)
        | FloatAttribute(0.0, 1.0, inclusive_maximum=True)
        | NoneAttribute()
    }

    def __init__(
        self,
        window_size: WINDOW_SIZE_TYPE,
        stride: int = 1,
        n_components: int | float = None,
        **kwargs,
    ):

        if n_components is not None:
            if not isinstance(n_components, (int, float)) or isinstance(
                n_components, bool
            ):
                raise TypeError("`n_components` should be integer or 'auto'")
            if isinstance(n_components, int) and n_components < 1:
                raise ValueError("`n_components` should be strictly positive")
            if isinstance(n_components, float) and (
                n_components <= 0 or n_components > 1
            ):
                raise ValueError("`n_components` between 0 and 1")

        self.n_components = n_components

        super().__init__(window_size, stride, **kwargs)

    def _initialize_detector(self, **kwargs) -> PCA:
        return PCA(n_components=self.n_components, **kwargs)

    def _supervision(self):
        return Supervision.SEMI_SUPERVISED
