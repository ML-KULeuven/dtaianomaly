from typing import Optional, Union

import numpy as np
import stumpy
from sklearn.exceptions import NotFittedError

from dtaianomaly import utils
from dtaianomaly.anomaly_detection.BaseDetector import BaseDetector, Supervision
from dtaianomaly.anomaly_detection.windowing_utils import (
    check_is_valid_window_size,
    compute_window_size,
    reverse_sliding_window,
)


class MatrixProfileDetector(BaseDetector):
    """
    Anomaly detector based on the Matrix Profile

    Use the STOMP algorithm to detect anomalies in a time series
    [Zhu2016matrixII]_. STOMP is a fast and scalable algorithm for computing
    the matrix profile, which measures the distance from each sequence to the
    most similar other sequence. Consequently, the matrix profile can be used
    to quantify how anomalous a subsequence is, because it has a large distance
    to all other subsequences.

    Parameters
    ----------
    window_size: int or str
        The window size to use for computing the matrix profile. This
        value will be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.
    normalize : bool, default=True
        Whether to z-normalize the time series before computing
        the matrix profile.
    p : float, default=2.0
        The norm to use for computing the matrix profile.
    k : int, default=1
        The k-th nearest neighbor to use for computing the sequence distance
        in the matrix profile.
    novelty: bool, default=False
        If novelty detection should be performed, i.e., detect anomalies in regard
        to the train time series. If False, the matrix profile equals a self-join,
        otherwise the matrix profile will be computed by comparing the subsequences
        in the test data to the subsequences in the train data.

    Attributes
    ----------
    window_size_: int
        The effectively used window size for computing the matrix profile
    X_reference_ : np.ndarray of shape (n_samples, n_attributes)
        The reference time series. Only available if ``novelty=True``

    Notes
    -----
    If the given time series is multivariate, the matrix profile is computed
    for each dimension separately and then summed up.

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import MatrixProfileDetector
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> matrix_profile = MatrixProfileDetector(window_size=50).fit(x)
    >>> matrix_profile.decision_function(x)
    array([1.20325439, 1.20690487, 1.20426043, ..., 1.47953858, 1.50188666,
           1.49891281])

    References
    ----------
    .. [Zhu2016matrixII] Y. Zhu et al., "Matrix Profile II: Exploiting a Novel
       Algorithm and GPUs to Break the One Hundred Million Barrier for Time Series
       Motifs and Joins," 2016 IEEE 16th International Conference on Data Mining
       (ICDM), Barcelona, Spain, 2016, pp. 739-748, doi: `10.1109/ICDM.2016.0085 <https://doi.org/10.1109/ICDM.2016.0085>`_.
    """

    window_size: Union[int, str]
    normalize: bool
    p: float
    k: int
    novelty: bool
    window_size_: int
    X_reference_: np.ndarray

    def __init__(
        self,
        window_size: Union[int, str],
        normalize: bool = True,
        p: float = 2.0,
        k: int = 1,
        novelty: bool = False,
    ) -> None:
        super().__init__(Supervision.UNSUPERVISED)

        check_is_valid_window_size(window_size)

        if not isinstance(normalize, bool):
            raise TypeError("`normalize` should be boolean")

        if not isinstance(p, (float, int)) or isinstance(p, bool):
            raise TypeError("`p` should be numeric")
        if p < 1.0:
            raise ValueError("`p` is a p-norm, value should be higher than 1.")

        if not isinstance(k, int) or isinstance(k, bool):
            raise TypeError("`k` should be integer")
        if k < 1:
            raise ValueError("`k` should be strictly positive")

        if not isinstance(novelty, bool):
            raise TypeError("'novelty' should be a boolean")

        self.window_size = window_size
        self.normalize = normalize
        self.p = p
        self.k = k
        self.novelty = novelty

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs
    ) -> "MatrixProfileDetector":
        """
        Fit this detector to the given data. Function is only present for
        consistency. Only saves the given data as a numpy array if
        ``novelty=True``.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_attributes)
            Input time series.
        y: ignored
            Not used, present for API consistency by convention.
        kwargs:
            Additional parameters to be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.

        Returns
        -------
        self: MatrixProfileDetector
            Returns the instance itself
        """
        if not utils.is_valid_array_like(X):
            raise ValueError("Input must be numerical array-like")

        X = np.asarray(X)
        self.window_size_ = compute_window_size(X, self.window_size, **kwargs)
        if self.novelty:
            self.X_reference_ = np.asarray(X)

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_attributes)
            Input time series.

        Returns
        -------
        matrix_profile: array-like of shape (n_samples)
            Matrix profile scores. Higher is more anomalous.

        Raises
        ------
        ValueError
            If `X` is not a valid array.
        NotFittedError
            If novelty detection must be performed (``novelty=True``), but this
            detector has not been fitted yet.
        ValueError
            If novelty detection must be performed (``novelty=True``), but the reference
            data ``X_reference_`` has a different number of attributes than the given
            data ``X``.
        """
        if not utils.is_valid_array_like(X):
            raise ValueError(f"Input must be numerical array-like")

        # Make sure X is a numpy array
        X = np.asarray(X)

        if not hasattr(self, "window_size_"):
            raise NotFittedError("Call the fit function before making predictions!")
        if self.novelty:
            nb_attributes_test = 1 if len(X.shape) == 1 else X.shape[1]
            nb_attributes_reference = (
                1 if len(self.X_reference_.shape) == 1 else self.X_reference_.shape[1]
            )
            if nb_attributes_reference != nb_attributes_test:
                raise ValueError(
                    f"Trying to detect anomalies with Matrix Profile using ``novelty=True``, but the number of attributes "
                    f"in the reference data is different from the number of attributes in the test data: "
                    f"({nb_attributes_reference} != {nb_attributes_test})!"
                )

        # Stumpy assumes arrays of shape [C T], where C is the number of "channels"
        # and T the number of time samples

        # This function works for multivariate and univariate signals
        ignore_trivial = True if not self.novelty else False
        if len(X.shape) == 1 or X.shape[1] == 1:
            T_B = None if not self.novelty else self.X_reference_.squeeze()
            matrix_profile = stumpy.stump(
                X.squeeze(),
                T_B=T_B,
                m=self.window_size_,
                normalize=self.normalize,
                p=self.p,
                k=self.k,
                ignore_trivial=ignore_trivial,
            )[
                :, self.k - 1
            ]  # Needed if k>1?
        else:
            if self.novelty:
                matrix_profiles = np.full(
                    shape=(X.shape[0] - self.window_size_ + 1, X.shape[1]),
                    fill_value=np.nan,
                )
                for attribute in range(X.shape[1]):
                    matrix_profiles[:, attribute] = stumpy.stump(
                        X[:, attribute],
                        T_B=self.X_reference_[:, attribute],
                        m=self.window_size_,
                        normalize=self.normalize,
                        p=self.p,
                        k=self.k,
                        ignore_trivial=ignore_trivial,
                    )[:, self.k - 1]
            else:
                matrix_profiles, _ = stumpy.mstump(
                    X.transpose(),
                    m=self.window_size_,
                    discords=True,
                    normalize=self.normalize,
                    p=self.p,
                )
            matrix_profile = np.sum(matrix_profiles, axis=0)

        return reverse_sliding_window(matrix_profile, self.window_size_, 1, X.shape[0])
