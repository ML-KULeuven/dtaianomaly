""" This function is adapted from TSB-AD """

import numpy as np
from typing import Optional, Union
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError

from dtaianomaly import utils
from dtaianomaly.anomaly_detection.BaseDetector import BaseDetector, Supervision
from dtaianomaly.anomaly_detection.windowing_utils import sliding_window, reverse_sliding_window, check_is_valid_window_size, compute_window_size


class RobustPrincipalComponentAnalysis(BaseDetector):
    """
    Anomaly detection based on Robust Principal Component Analysis (Robust PCA).

    Assume that the data matrix is a superposition of a low-rank component and a s
    parse component. Robust PCA [Candes2011robust]_ will solve this decomposition
    as a convex optimization problem. The superposition offers a principeled manner
    to robust PCA, since the methodology can recover the principal components (first
    component) of a data matrix even though a positive fraction of the entries are
    arbitrarly corrupted or anomalous (second component).

    Notes
    -----
    In most existing implementations, Robust PCA only takes one observation at a
    time into account (i.e., does not look at windows). However, Robust PCA can
    not be applied to a single variable, which is the case for univariate data.
    Therefore, we added a parameter ``window_size`` to apply Robust PCA in windows
    of a univariate time series, to make it applicable. Common behavior on multivariate
    time series can be obtained by setting ``window_size = 1``.

    Parameters
    ----------
    window_size: int or str
        The window size to use for extracting sliding windows from the time series. This
        value will be passed to :py:meth:`~dtaianomaly.anomaly_detection.compute_window_size`.
    stride: int, default=1
        The stride, i.e., the step size for extracting sliding windows from the time series.
    max_iter: int, default=1000
        The maximum number of iterations allowed to optimize the low rank approximation.
    kwargs:
        Additional parameters to be passed PCA of Sklearn.

    Attributes
    ----------
    window_size_: int
        The effectively used window size for this anomaly detector
    pca_ : PCA
        The PCA-object used to project the data in a lower dimension.

    Examples
    --------
    >>> from dtaianomaly.anomaly_detection import RobustPrincipalComponentAnalysis
    >>> from dtaianomaly.data import demonstration_time_series
    >>> x, y = demonstration_time_series()
    >>> rpca = RobustPrincipalComponentAnalysis(2).fit(x)
    >>> rpca.decision_function(x)
    array([1.28436687, 1.29156655, 1.33793287, ..., 1.35563558, 1.25948662,
           1.2923824 ])

    References
    ----------
    .. [Candes2011robust] Emmanuel J. Candès, Xiaodong Li, Yi Ma, and John Wright. 2011. Robust principal
       component analysis? J. ACM 58, 3, Article 11 (June 2011), 37 pages. doi: `10.1145/1970392.1970395 <https://doi.org/10.1145/1970392.1970395>`_
    """
    window_size: Union[int, str]
    stride: int
    max_iter: int
    kwargs: dict
    window_size_: int
    pca_: PCA

    def __init__(self, window_size: Union[str, int], stride: int = 1, max_iter: int = 1000, **kwargs):
        super().__init__(Supervision.SEMI_SUPERVISED)

        check_is_valid_window_size(window_size)

        if not isinstance(max_iter, int) or isinstance(max_iter, bool):
            raise TypeError("`max_iter` should be an integer")
        if max_iter < 1:
            raise ValueError("`max_iter`should be at least 1")

        PCA(n_components=0.1, **kwargs)  # Check if PCA can be initialized

        self.window_size = window_size
        self.stride = stride
        self.max_iter = max_iter
        self.kwargs = kwargs

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> 'RobustPrincipalComponentAnalysis':
        """
        Fit this Robust PCA to the given data

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
        self: RobustPrincipleComponentAnalysis
            Returns the instance itself

        Raises
        ------
        ValueError
            If `X` is not a valid array.
        """
        if not utils.is_valid_array_like(X):
            raise ValueError("Input must be numerical array-like")

        # Make sure X is a numpy array
        X = np.asarray(X)

        # Compute the windows
        self.window_size_ = compute_window_size(X, self.window_size, **kwargs)
        sliding_windows = sliding_window(X, self.window_size_, self.stride)

        # Apply robust PCA
        robust_pca = _RobustPCA(sliding_windows)
        L, S = robust_pca.fit(max_iter=self.max_iter)
        self.pca_ = PCA(n_components=L.shape[1], **self.kwargs)
        self.pca_.fit(L)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, 'pca_'):
            raise NotFittedError('Call the fit function before making predictions!')
        if not utils.is_valid_array_like(X):
            raise ValueError(f"Input must be numerical array-like")

        # Make sure X is a numpy array
        X = np.asarray(X)

        # Convert to sliding windows
        windows = sliding_window(X, self.window_size_, self.stride)

        # DO RPCA
        L = self.pca_.transform(windows)
        S = np.absolute(windows - L)
        per_window_decision_scores = S.sum(axis=1)

        # Get an anomaly score for each window
        decision_scores = reverse_sliding_window(per_window_decision_scores, self.window_size_, self.stride, X.shape[0])

        return decision_scores


# From https://github.com/dganguli/robust-pca
class _RobustPCA:

    def __init__(self, D, mu=None, lmbda=None):
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)

        if mu:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / (4 * np.linalg.norm(self.D, ord=1))

        self.mu_inv = 1 / self.mu

        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))

    @staticmethod
    def frobenius_norm(M):
        return np.linalg.norm(M, ord='fro')

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit(self, max_iter=1000):
        iter = 0
        err = np.inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)

        _tol = 1E-7 * self.frobenius_norm(self.D)

        # this loop implements the principal component pursuit (PCP) algorithm
        # located in the table on page 29 of https://arxiv.org/pdf/0912.3599.pdf
        while (err > _tol) and iter < max_iter:
            Lk = self.svd_threshold(self.D - Sk + self.mu_inv * Yk, self.mu_inv)  # this line implements step 3
            Sk = self.shrink(self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)  # this line implements step 4
            Yk = Yk + self.mu * (self.D - Lk - Sk)  # this line implements step 5
            err = self.frobenius_norm(self.D - Lk - Sk)
            iter += 1

        self.L = Lk
        self.S = Sk
        return Lk, Sk
