from abc import ABC
from typing import Optional, Tuple

import numpy as np

from dtaianomaly.evaluation.metrics import ProbaMetric

# TODO docs -> Maybe create subsections on the webpage? Or add something similar to anomaly detectors?
# TODO testing -> Maybe we can include the examples from tsb ad paper (fig 5)
# TODO checks on input in __init__ (of VUS as well!)


class RangeAucMetric(ProbaMetric, ABC):
    """Base class for range-based area under the curve metrics.

    All range-based metrics support continuous scorings and share a common implementation of the confusion matrix.
    See the subclasses' documentation for an explanation of the corresponding metric.

    .. note::

       These implementations are adapted from TimeEval [ref] (add ref in the note?)
    """

    buffer_size: Optional[int]
    compatibility_mode: bool
    max_samples: int

    def __init__(
        self,
        buffer_size: Optional[int] = None,
        compatibility_mode: bool = False,
        max_samples: int = 250,
    ):
        self.buffer_size = buffer_size
        self.compatibility_mode = compatibility_mode
        self.max_samples = max_samples

    @staticmethod
    def _anomaly_bounds(y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """corresponds to range_convers_new"""
        # convert to boolean/binary
        labels = y_true > 0
        # deal with start and end of time series
        labels = np.diff(np.r_[0, labels, 0])
        # extract begin and end of anomalous regions
        index = np.arange(0, labels.shape[0])
        starts = index[labels == 1]
        ends = index[labels == -1]
        return starts, ends

    def _extend_anomaly_labels(
        self, y_true: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extends the anomaly labels with slopes on both ends. Makes the labels continuous instead of binary."""
        starts, ends = self._anomaly_bounds(y_true)

        if self.buffer_size is None:
            # per default: set buffer size as median anomaly length:
            self.buffer_size = int(np.median(ends - starts))

        if self.buffer_size <= 1:
            if self.compatibility_mode:
                anomalies = np.array(list(zip(starts, ends - 1)))
            else:
                anomalies = np.array(list(zip(starts, ends)))
            return y_true.astype(float), anomalies

        y_true_cont = y_true.astype(float)
        slope_length = self.buffer_size // 2
        length = y_true_cont.shape[0]
        if self.compatibility_mode:
            for i, (s, e) in enumerate(zip(starts, ends)):
                e -= 1
                x1 = np.arange(e, min(e + slope_length, length))
                y_true_cont[x1] += np.sqrt(1 - (x1 - e) / self.buffer_size)
                x2 = np.arange(max(s - slope_length, 0), s)
                y_true_cont[x2] += np.sqrt(1 - (s - x2) / self.buffer_size)
            y_true_cont = np.clip(y_true_cont, 0, 1)
            starts, ends = self._anomaly_bounds(y_true_cont)
            anomalies = np.array(list(zip(starts, ends - 1)))

        else:
            slope = np.linspace(1 / np.sqrt(2), 1, slope_length + 1)
            anomalies = np.empty((starts.shape[0], 2), dtype=np.int_)
            for i, (s, e) in enumerate(zip(starts, ends)):
                s0 = max(0, s - slope_length)
                s1 = s + 1
                y_true_cont[s0:s1] = np.maximum(slope[s0 - s1 :], y_true_cont[s0:s1])
                e0 = e - 1
                e1 = min(length, e + slope_length)
                y_true_cont[e0:e1] = np.maximum(
                    slope[e0 - e1 :][::-1], y_true_cont[e0:e1]
                )
                anomalies[i] = [s0, e1]
        return y_true_cont, anomalies

    def _uniform_threshold_sampling(self, y_score: np.ndarray) -> np.ndarray:
        if self.compatibility_mode:
            n_samples = 250
        else:
            n_samples = min(self.max_samples, y_score.shape[0])
        thresholds: np.ndarray = np.sort(y_score)[::-1]
        thresholds = thresholds[
            np.linspace(0, thresholds.shape[0] - 1, n_samples, dtype=np.int_)
        ]
        return thresholds

    def _range_pr_roc_auc_support(
        self, y_true: np.ndarray, y_score: np.ndarray
    ) -> Tuple[float, float]:
        y_true_cont, anomalies = self._extend_anomaly_labels(y_true)
        thresholds = self._uniform_threshold_sampling(y_score)
        p = np.average([np.sum(y_true), np.sum(y_true_cont)])

        recalls = np.zeros(thresholds.shape[0] + 2)  # tprs
        fprs = np.zeros(thresholds.shape[0] + 2)
        precisions = np.ones(thresholds.shape[0] + 1)

        for i, t in enumerate(thresholds):
            y_pred = y_score >= t
            product = y_true_cont * y_pred
            tp = np.sum(product)
            # fp = np.dot((np.ones_like(y_pred) - y_true_cont).T, y_pred)
            fp = np.sum(y_pred) - tp
            n = len(y_pred) - p

            existence_reward = [np.sum(product[s : e + 1]) > 0 for s, e in anomalies]
            existence_reward = np.sum(existence_reward) / anomalies.shape[0]

            recall = min(tp / p, 1) * existence_reward  # = tpr
            fpr = min(fp / n, 1)
            precision = tp / np.sum(y_pred)

            recalls[i + 1] = recall
            fprs[i + 1] = fpr
            precisions[i + 1] = precision

        recalls[-1] = 1
        fprs[-1] = 1

        range_pr_auc: float = np.sum(
            (recalls[1:-1] - recalls[:-2]) * (precisions[1:] + precisions[:-1]) / 2
        )
        range_roc_auc: float = np.sum(
            (fprs[1:] - fprs[:-1]) * (recalls[1:] + recalls[:-1]) / 2
        )

        return range_pr_auc, range_roc_auc


class RangeAreaUnderPR(RangeAucMetric):
    """Computes the area under the precision-recall-curve using the range-based precision and range-based recall
    definition from Paparrizos et al. published at VLDB 2022 [PaparrizosEtAl2022]_.

    We first extend the anomaly labels by two slopes of ``buffer_size//2`` length on both sides of each anomaly,
    uniformly sample thresholds from the anomaly score, and then compute the confusion matrix for all thresholds.
    Using the resulting precision and recall values, we can plot a curve and compute its area.

    We make some changes to the original implementation from [PaparrizosEtAl2022]_ because we do not agree with the
    original assumptions. To reproduce the original results, you can set the parameter ``compatibility_mode=True``. This
    will compute exactly the same values as the code by the authors of the paper.

    The following things are different in TimeEval compared to the original version:

    - For the recall (FPR) existence reward, we count anomalies as separate events, even if the added slopes overlap.
    - Overlapping slopes don’t sum up in their anomaly weight, but we just take to maximum anomaly weight for each
      point in the ground truth.
    - The original slopes are asymmetric: The slopes at the end of anomalies are a single point shorter than the ones at
      the beginning of anomalies. We use symmetric slopes of the same size for the beginning and end of anomalies.
    - We use a linear approximation of the slopes instead of the convex slope shape presented in the paper.

    Parameters
    ----------
    buffer_size: int, default=None
        Size of the buffer region around an anomaly. We add an increasing slope of size ``buffer_size//2`` to the
        beginning of anomalies and a decreasing slope of size ``buffer_size//2`` to the end of anomalies. Per default
        (when ``buffer_size==None``), ``buffer_size`` is the median length of the anomalies within the time series.
        However, you can also set it to the period size of the dominant frequency or any other desired value.
    compatibility_mode: bool, default=False
        When set to ``True``, produces exactly the same output as the metric implementation by the original authors.
        Otherwise, TimeEval uses a slightly improved implementation that fixes some bugs and uses linear slopes.
    max_samples: int, default=250
        Calculating precision and recall for many thresholds is quite slow. We, therefore, uniformly sample thresholds
        from the available score space. This parameter controls the maximum number of thresholds; too low numbers
        degrade the metrics' quality.
    """

    def __init__(
        self,
        buffer_size: Optional[int] = None,
        compatibility_mode: bool = False,
        max_samples: int = 250,
    ):
        super().__init__(buffer_size, compatibility_mode, max_samples)

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        range_pr_auc, _ = self._range_pr_roc_auc_support(y_true, y_pred)
        return range_pr_auc


class RangeAreaUnderROC(RangeAucMetric):
    """Computes the area under the receiver-operating-characteristic-curve using the range-based TPR and
    range-based FPR definition from Paparrizos et al. published at VLDB 2022 [PaparrizosEtAl2022]_.

    We first extend the anomaly labels by two slopes of ``buffer_size//2`` length on both sides of each anomaly,
    uniformly sample thresholds from the anomaly score, and then compute the confusion matrix for all thresholds.
    Using the resulting false positive (FPR) and false positive rates (FPR), we can plot a curve and compute its area.

    We make some changes to the original implementation from [PaparrizosEtAl2022]_ because we do not agree with the
    original assumptions. To reproduce the original results, you can set the parameter ``compatibility_mode=True``. This
    will compute exactly the same values as the code by the authors of the paper.

    The following things are different in TimeEval compared to the original version:

    - For the recall (FPR) existence reward, we count anomalies as separate events, even if the added slopes overlap.
    - Overlapping slopes don’t sum up in their anomaly weight, but we just take to maximum anomaly weight for each
      point in the ground truth.
    - The original slopes are asymmetric: The slopes at the end of anomalies are a single point shorter than the ones at
      the beginning of anomalies. We use symmetric slopes of the same size for the beginning and end of anomalies.
    - We use a linear approximation of the slopes instead of the convex slope shape presented in the paper.

    Parameters
    ----------
    buffer_size: int, default=None
        Size of the buffer region around an anomaly. We add an increasing slope of size ``buffer_size//2`` to the
        beginning of anomalies and a decreasing slope of size ``buffer_size//2`` to the end of anomalies. Per default
        (when ``buffer_size==None``), ``buffer_size`` is the median length of the anomalies within the time series.
        However, you can also set it to the period size of the dominant frequency or any other desired value.
    compatibility_mode: bool, default=False
        When set to ``True``, produces exactly the same output as the metric implementation by the original authors.
        Otherwise, TimeEval uses a slightly improved implementation that fixes some bugs and uses linear slopes.
    max_samples: int, default= 250
        Calculating precision and recall for many thresholds is quite slow. We, therefore, uniformly sample thresholds
        from the available score space. This parameter controls the maximum number of thresholds; too low numbers
        degrade the metrics' quality.
    """

    def __init__(
        self,
        buffer_size: Optional[int] = None,
        compatibility_mode: bool = False,
        max_samples: int = 250,
    ):
        super().__init__(buffer_size, compatibility_mode, max_samples)

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        _, range_auc_roc = self._range_pr_roc_auc_support(y_true, y_pred)
        return range_auc_roc


class VolumeUnderPR(RangeAucMetric):
    """Computes the volume under the precision-recall-buffer_size-surface using the range-based precision and
    range-based recall definition from Paparrizos et al. published at VLDB 2022 [PaparrizosEtAl2022]_.

    For all buffer sizes from 0 to ``max_buffer_size``, we first extend the anomaly labels by two slopes of
    ``buffer_size//2`` length on both sides of each anomaly, uniformly sample thresholds from the anomaly score, and
    then compute the confusion matrix for all thresholds. Using the resulting precision and recall values, we can plot
    a curve and compute its area.

    This metric includes similar changes as :class:`~timeeval.metrics.RangePrAUC`, which can be disabled using the
    ``compatibility_mode`` parameter.

    Parameters
    ----------
    max_buffer_size: int, default=500
        Maximum size of the buffer region around an anomaly. We iterate over all buffer sizes from 0 to
        ``may_buffer_size`` to create the surface.
    compatibility_mode: bool, default=False
        When set to ``True``, produces exactly the same output as the metric implementation by the original authors.
        Otherwise, TimeEval uses a slightly improved implementation that fixes some bugs and uses linear slopes.
    max_samples: int, default=250
        Calculating precision and recall for many thresholds is quite slow. We, therefore, uniformly sample thresholds
        from the available score space. This parameter controls the maximum number of thresholds; too low numbers
        degrade the metrics' quality.
    """

    max_buffer_size: int

    def __init__(
        self,
        max_buffer_size: int = 500,
        compatibility_mode: bool = False,
        max_samples: int = 250,
    ):
        super().__init__(None, compatibility_mode, max_samples)
        self.max_buffer_size = max_buffer_size

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        prs = np.zeros(self.max_buffer_size + 1)
        for bs in np.arange(0, self.max_buffer_size + 1):
            self.buffer_size = bs
            pr_auc, _ = self._range_pr_roc_auc_support(y_true, y_pred)
            prs[bs] = pr_auc
        range_pr_volume: float = np.sum(prs) / (self.max_buffer_size + 1)
        return range_pr_volume


class VolumeUnderROC(RangeAucMetric):
    """Computes the volume under the receiver-operating-characteristic-buffer_size-surface using the range-based TPR and
    range-based FPR definition from Paparrizos et al. published at VLDB 2022 [PaparrizosEtAl2022]_.

    For all buffer sizes from 0 to ``max_buffer_size``, we first extend the anomaly labels by two slopes of
    ``buffer_size//2`` length on both sides of each anomaly, uniformly sample thresholds from the anomaly score, and
    then compute the confusion matrix for all thresholds. Using the resulting false positive (FPR) and false positive
    rates (FPR), we can plot a curve and compute its area.

    This metric includes similar changes as :class:`~timeeval.metrics.RangeRocAUC`, which can be disabled using the
    ``compatibility_mode`` parameter.

    Parameters
    ----------
    max_buffer_size: int, default=500
        Maximum size of the buffer region around an anomaly. We iterate over all buffer sizes from 0 to
        ``may_buffer_size`` to create the surface.
    compatibility_mode: bool, default=False
        When set to ``True``, produces exactly the same output as the metric implementation by the original authors.
        Otherwise, TimeEval uses a slightly improved implementation that fixes some bugs and uses linear slopes.
    max_samples: int, default=250
        Calculating precision and recall for many thresholds is quite slow. We, therefore, uniformly sample thresholds
        from the available score space. This parameter controls the maximum number of thresholds; too low numbers
        degrade the metrics' quality.
    """

    max_buffer_size: int

    def __init__(
        self,
        max_buffer_size: int = 500,
        compatibility_mode: bool = False,
        max_samples: int = 250,
    ):
        super().__init__(None, compatibility_mode, max_samples)
        self.max_buffer_size = max_buffer_size

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        rocs = np.zeros(self.max_buffer_size + 1)
        for bs in np.arange(0, self.max_buffer_size + 1):
            self.buffer_size = bs
            _, roc_auc = self._range_pr_roc_auc_support(y_true, y_pred)
            rocs[bs] = roc_auc
        range_pr_volume: float = np.sum(rocs) / (self.max_buffer_size + 1)
        return range_pr_volume
