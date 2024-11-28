""" This function is adapted from TSB-UAD """

import numpy as np
from typing import Optional
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError

from dtaianomaly import utils
from dtaianomaly.anomaly_detection.BaseDetector import BaseDetector, Supervision


class RobustPrincipleComponentAnalysis(BaseDetector):
    """
    TODO documentation
    """
    max_iter: int
    zero_pruning: bool
    pca_: PCA

    def __init__(self, max_iter: int = 1000, zero_pruning: bool = True):
        super().__init__(Supervision.SEMI_SUPERVISED)
        # TODO allow for a window size? For univariate time series? -> if yes, add the RobustPCA to the checks with automatic window size
        if not isinstance(max_iter, int) or isinstance(max_iter, bool):
            raise TypeError("`max_iter` should be an integer")
        if max_iter < 1:
            raise ValueError("`max_iter`should be at least 1")

        if not isinstance(zero_pruning, bool):
            raise TypeError("`zero_pruning` should be boolean")

        self.max_iter = max_iter
        self.zero_pruning = zero_pruning

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'RobustPrincipleComponentAnalysis':
        if not utils.is_valid_array_like(X):
            raise ValueError("Input must be numerical array-like")

        if self.zero_pruning:
            non_zero_columns = np.any(X != 0, axis=0)
            X = X[:, non_zero_columns]

        robust_pca = _RobustPCA(X)
        L, S = robust_pca.fit(max_iter=self.max_iter)
        self.pca_ = PCA(n_components=L.shape[1])
        self.pca_.fit(L)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, 'pca_'):
            raise NotFittedError('Call the fit function before making predictions!')
        if not utils.is_valid_array_like(X):
            raise ValueError(f"Input must be numerical array-like")

        X = np.asarray(X)
        # TODO is this correct? Shouldn't we use the robust PCA object here?
        L = self.pca_.transform(X)
        S = np.absolute(X - L)
        return S.sum(axis=1)


class _RobustPCA:

    # Based on https://github.com/dganguli/robust-pca/blob/master/r_pca.py
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
