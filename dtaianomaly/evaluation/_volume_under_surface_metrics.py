from abc import ABC

import numba as nb
import numpy as np

from dtaianomaly.evaluation._ProbaMetric import ProbaMetric
from dtaianomaly.type_validation import IntegerAttribute, NoneAttribute

__all__ = ["RangeAreaUnderROC", "RangeAreaUnderPR", "VolumeUnderROC", "VolumeUnderPR"]

_DEFAULT_BUFFER_SIZE: int | None = None
_DEFAULT_MAX_BUFFER_SIZE: int = 100
_DEFAULT_MAX_SAMPLES: int = 250


class RangeAucMetric(ProbaMetric, ABC):
    """
    Base class for range-based area under the curve metrics, to share common implementations of
    the confusion matrix.
    """

    max_samples: int

    attribute_validation = {"max_samples": IntegerAttribute(minimum=1)}

    def __init__(self, max_samples: int = _DEFAULT_MAX_SAMPLES):
        self.max_samples = max_samples

    @staticmethod
    def _anomaly_bounds(y_true: np.ndarray) -> (np.ndarray, np.ndarray):
        """Returns the start and (inclusive) end of each anomalous event."""
        labels = np.diff(np.r_[0, y_true > 0, 0].astype(np.int8))
        starts = np.flatnonzero(labels == 1)
        ends = np.flatnonzero(labels == -1) - 1
        return starts, ends

    def _extend_anomaly_labels(
        self, y_true: np.ndarray, buffer_size: int | None
    ) -> (np.ndarray, np.ndarray):
        """
        Extends the anomaly labels with slopes on both ends, which makes the labels continuous
        instead of binary. Returns the continuous labels and the segments (with inclusive ends)
        used for the existence reward. Corresponds to 'sequencing' and 'new_sequence' in TSB-AD.
        """
        starts, ends = self._anomaly_bounds(y_true)

        if buffer_size is None:
            # per default: set buffer size as median anomaly length:
            buffer_size = int(np.median(ends - starts + 1))

        y_true_cont = (y_true > 0).astype(float)
        length = y_true_cont.shape[0]
        slope_length = buffer_size // 2

        if slope_length > 0:
            for s, e in zip(starts, ends):
                x1 = np.arange(e + 1, min(e + slope_length + 1, length))
                y_true_cont[x1] += np.sqrt(1 - (x1 - e) / buffer_size)
                x2 = np.arange(max(s - slope_length, 0), s)
                y_true_cont[x2] += np.sqrt(1 - (s - x2) / buffer_size)
            y_true_cont = np.minimum(y_true_cont, 1)

        # Merge the anomalies of which the extended regions overlap
        separated = ends[:-1] + slope_length < starts[1:] - slope_length
        segment_starts = np.r_[
            max(starts[0] - slope_length, 0), starts[1:][separated] - slope_length
        ]
        segment_ends = np.r_[
            ends[:-1][separated] + slope_length,
            min(ends[-1] + slope_length, length - 1),
        ]
        segments = np.column_stack((segment_starts, segment_ends)).astype(np.int64)
        return y_true_cont, segments.reshape(-1, 2)

    def _uniform_threshold_sampling(
        self, y_score: np.ndarray
    ) -> (np.ndarray, np.ndarray):
        """Returns the sampled thresholds and the number of predicted anomalies for each threshold."""
        sorted_scores = np.sort(y_score)
        thresholds = sorted_scores[::-1][
            np.linspace(0, sorted_scores.shape[0] - 1, self.max_samples, dtype=np.int_)
        ]
        n_pred = sorted_scores.shape[0] - np.searchsorted(
            sorted_scores, thresholds, side="left"
        )
        return thresholds, n_pred.astype(float)

    def _range_pr_roc_auc_support(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        buffer_size: int | None,
        thresholds: np.ndarray,
        n_pred: np.ndarray,
    ) -> (float, float):
        y_true_cont, segments = self._extend_anomaly_labels(y_true, buffer_size)

        # Only the observations with a non-zero label contribute to the true positives
        relevant = np.flatnonzero(y_true_cont > 0)
        labels = y_true_cont[relevant]
        # The weight of the slopes, to compute the number of positives for each threshold
        slopes = np.where(y_true[relevant] > 0, 0.0, labels)
        segment_max = np.array([y_score[s : e + 1].max() for s, e in segments])

        recalls, fprs, precisions = _range_curves(
            thresholds,
            n_pred,
            y_score[relevant],
            labels,
            slopes,
            segment_max,
            float(np.sum(y_true)),
            y_true.shape[0],
        )

        range_roc_auc = np.sum(
            (fprs[1:] - fprs[:-1]) * (recalls[1:] + recalls[:-1]) / 2
        )
        range_pr_auc = np.sum((recalls[1:-1] - recalls[:-2]) * precisions[1:])
        return float(range_pr_auc), float(range_roc_auc)


@nb.njit(fastmath=True, cache=True, parallel=True)
def _range_curves(
    thresholds: np.ndarray,
    n_pred: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    slopes: np.ndarray,
    segment_max: np.ndarray,
    p: float,
    n: int,
):
    recalls = np.zeros(thresholds.shape[0] + 2)  # tprs
    fprs = np.zeros(thresholds.shape[0] + 2)
    precisions = np.ones(thresholds.shape[0] + 1)

    for i in nb.prange(thresholds.shape[0]):
        t = thresholds[i]
        tp = 0.0
        predicted_slopes = 0.0
        for k in range(scores.shape[0]):
            if scores[k] >= t:
                tp += labels[k]
                predicted_slopes += slopes[k]
        existence_reward = np.sum(segment_max >= t) / segment_max.shape[0]

        # The number of positives depends on the predicted part of the slopes
        p_new = p + predicted_slopes / 2
        n_new = n - p_new

        recalls[i + 1] = min(tp / p_new, 1.0) * existence_reward
        fprs[i + 1] = (n_pred[i] - tp) / n_new if n_new > 0 else 0.0
        precisions[i + 1] = tp / n_pred[i]

    recalls[-1] = 1
    fprs[-1] = 1

    return recalls, fprs, precisions


class RangeAreaUnderPR(RangeAucMetric):
    """
    Computes the area under the range-based PR-curve :cite:`paparrizos2022volume,boniol2025vus`.

    A slope of length ``buffer_size // 2`` is added at the beginning and end of each anomalous
    event. Next, the precision and recall is computed, taking into account the slopes in ground
    truth labels to allow for some small misalignment in the predicted and actual anomalous events.
    Then, ``max_samples`` thresholds are sampled uniformly from the anomaly scores to compute
    the new precision and recall, after which the area under the curve can be computed as final
    evaluation score.

    Parameters
    ----------
    buffer_size : int, default=None
        Size of the buffer region around an anomaly. We add an increasing slope of size ``buffer_size//2`` to the
        beginning of anomalies and a decreasing slope of size ``buffer_size//2`` to the end of anomalies. Per default
        (when ``buffer_size==None``), ``buffer_size`` is the median length of the anomalies within the time series.
        However, you can also set it to the period size of the dominant frequency or any other desired value.
    max_samples : int, default=250
        Calculating precision and recall for many thresholds is quite slow. We, therefore, uniformly sample thresholds
        from the available score space. This parameter controls the number of thresholds; too low numbers
        degrade the metrics' quality. Use the default value to obtain results comparable with TSB-AD.

    See Also
    --------
    RangeAreaUnderROC: Compute the area under the range-based ROC-curve.
    VolumeUnderROC: Compute the volume under the range-based ROC-surface.
    VolumeUnderPR: Compute the volume under the range-based PR-surface.

    Examples
    --------
    >>> from dtaianomaly.evaluation import RangeAreaUnderPR
    >>> metric = RangeAreaUnderPR()
    >>> y_true = [0, 0, 0, 1, 1, 0, 0, 0]
    >>> y_pred = [1, 0, 0, 1, 1, 1, 0, 0]
    >>> metric.compute(y_true, y_pred)  # doctest: +ELLIPSIS
    0.676...
    """

    buffer_size: int | None
    attribute_validation = {
        "buffer_size": IntegerAttribute(minimum=1) | NoneAttribute()
    }

    def __init__(
        self,
        buffer_size: int = _DEFAULT_BUFFER_SIZE,
        max_samples: int = _DEFAULT_MAX_SAMPLES,
    ):
        super().__init__(max_samples)
        self.buffer_size = buffer_size

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        thresholds, n_pred = self._uniform_threshold_sampling(y_pred)
        range_pr_auc, _ = self._range_pr_roc_auc_support(
            y_true, y_pred, self.buffer_size, thresholds, n_pred
        )
        return range_pr_auc


class RangeAreaUnderROC(RangeAucMetric):
    """
    Computes the area under the range-based ROC-curve :cite:`paparrizos2022volume,boniol2025vus`.

    A slope of length ``buffer_size // 2`` is added at the beginning and end of each anomalous
    event. Next, the false positive rate and true positive rate is computed, taking into account
    the slopes in ground truth labels to allow for some small misalignment in the predicted and
    actual anomalous events. Then, ``max_samples`` thresholds are sampled uniformly from the
    anomaly scores to compute the new FPR and TPR, after which the area under the curve
    can be computed as final evaluation score.

    Parameters
    ----------
    buffer_size : int, default=None
        Size of the buffer region around an anomaly. We add an increasing slope of size ``buffer_size//2`` to the
        beginning of anomalies and a decreasing slope of size ``buffer_size//2`` to the end of anomalies. Per default
        (when ``buffer_size==None``), ``buffer_size`` is the median length of the anomalies within the time series.
        However, you can also set it to the period size of the dominant frequency or any other desired value.
    max_samples : int, default=250
        Calculating precision and recall for many thresholds is quite slow. We, therefore, uniformly sample thresholds
        from the available score space. This parameter controls the number of thresholds; too low numbers
        degrade the metrics' quality. Use the default value to obtain results comparable with TSB-AD.

    See Also
    --------
    RangeAreaUnderPR: Compute the area under the range-based PR-curve.
    VolumeUnderROC: Compute the volume under the range-based ROC-surface.
    VolumeUnderPR: Compute the volume under the range-based PR-surface.

    Examples
    --------
    >>> from dtaianomaly.evaluation import RangeAreaUnderROC
    >>> metric = RangeAreaUnderROC()
    >>> y_true = [0, 0, 0, 1, 1, 0, 0, 0]
    >>> y_pred = [1, 0, 0, 1, 1, 1, 0, 0]
    >>> metric.compute(y_true, y_pred)  # doctest: +ELLIPSIS
    0.885...
    """

    buffer_size: int | None
    attribute_validation = {
        "buffer_size": IntegerAttribute(minimum=1) | NoneAttribute()
    }

    def __init__(
        self,
        buffer_size: int = _DEFAULT_BUFFER_SIZE,
        max_samples: int = _DEFAULT_MAX_SAMPLES,
    ):
        super().__init__(max_samples)
        self.buffer_size = buffer_size

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        thresholds, n_pred = self._uniform_threshold_sampling(y_pred)
        _, range_auc_roc = self._range_pr_roc_auc_support(
            y_true, y_pred, self.buffer_size, thresholds, n_pred
        )
        return range_auc_roc


class VolumeUnderPR(RangeAucMetric):
    """
    Computes the volume under the range-based precision-recall-curve :cite:`paparrizos2022volume,boniol2025vus`.

    Create a buffer around the anomalous event (similar as for :py:class:`~dtaianomaly.evaluation.RangeAreaUnderPR`)
    for each buffer size in the range ``[0, max_buffer_size]``. Then, ``max_samples`` thresholds are
    sampled uniformly from the anomaly scores to compute the new precision and recall for each buffer size.
    Also varying the buffer size results in a volume (instead of a curve), and the final evaluation
    score is computed as the volume under this surface.

    Parameters
    ----------
    max_buffer_size : int, default=100
        Maximum size of the buffer region around an anomaly. We iterate over all buffer sizes from 0 to
        ``max_buffer_size`` to create the surface. This corresponds to the ``slidingWindow`` in TSB-AD.
    max_samples : int, default=250
        Calculating precision and recall for many thresholds is quite slow. We, therefore, uniformly sample thresholds
        from the available score space. This parameter controls the number of thresholds; too low numbers
        degrade the metrics' quality. Use the default value to obtain results comparable with TSB-AD.

    See Also
    --------
    RangeAreaUnderROC: Compute the area under the range-based ROC-curve.
    RangeAreaUnderPR: Compute the area under the range-based PR-curve.
    VolumeUnderROC: Compute the volume under the range-based ROC-surface.

    Examples
    --------
    >>> from dtaianomaly.evaluation import VolumeUnderPR
    >>> metric = VolumeUnderPR()
    >>> y_true = [0, 0, 0, 1, 1, 0, 0, 0]
    >>> y_pred = [1, 0, 0, 1, 1, 1, 0, 0]
    >>> metric.compute(y_true, y_pred)  # doctest: +ELLIPSIS
    0.963...
    """

    max_buffer_size: int
    attribute_validation = {"max_buffer_size": IntegerAttribute(minimum=1)}

    def __init__(
        self,
        max_buffer_size: int = _DEFAULT_MAX_BUFFER_SIZE,
        max_samples: int = _DEFAULT_MAX_SAMPLES,
    ):
        super().__init__(max_samples)
        self.max_buffer_size = max_buffer_size

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        thresholds, n_pred = self._uniform_threshold_sampling(y_pred)
        prs = np.zeros(self.max_buffer_size + 1)
        for buffer_size in range(self.max_buffer_size + 1):
            prs[buffer_size], _ = self._range_pr_roc_auc_support(
                y_true, y_pred, buffer_size, thresholds, n_pred
            )
        return float(np.mean(prs))


class VolumeUnderROC(RangeAucMetric):
    """
    Computes the volume under the range-based ROC-curve :cite:`paparrizos2022volume,boniol2025vus`.

    Create a buffer around the anomalous event (similar as for :py:class:`~dtaianomaly.evaluation.RangeAreaUnderROC`)
    for each buffer size in the range ``[0, max_buffer_size]``. Then, ``max_samples`` thresholds are
    sampled uniformly from the anomaly scores to compute the new FPR and TPR for each buffer size.
    Also varying the buffer size results in a volume (instead of a curve), and the final evaluation
    score is computed as the volume under this surface.

    Parameters
    ----------
    max_buffer_size : int, default=100
        Maximum size of the buffer region around an anomaly. We iterate over all buffer sizes from 0 to
        ``max_buffer_size`` to create the surface. This corresponds to the ``slidingWindow`` in TSB-AD.
    max_samples : int, default=250
        Calculating precision and recall for many thresholds is quite slow. We, therefore, uniformly sample thresholds
        from the available score space. This parameter controls the number of thresholds; too low numbers
        degrade the metrics' quality. Use the default value to obtain results comparable with TSB-AD.

    See Also
    --------
    RangeAreaUnderROC: Compute the area under the range-based ROC-curve.
    RangeAreaUnderPR: Compute the area under the range-based PR-curve.
    VolumeUnderPR: Compute the volume under the range-based PR-surface.

    Examples
    --------
    >>> from dtaianomaly.evaluation import VolumeUnderROC
    >>> metric = VolumeUnderROC()
    >>> y_true = [0, 0, 0, 1, 1, 0, 0, 0]
    >>> y_pred = [1, 0, 0, 1, 1, 1, 0, 0]
    >>> metric.compute(y_true, y_pred)  # doctest: +ELLIPSIS
    0.986...
    """

    max_buffer_size: int
    attribute_validation = {"max_buffer_size": IntegerAttribute(minimum=1)}

    def __init__(
        self,
        max_buffer_size: int = _DEFAULT_MAX_BUFFER_SIZE,
        max_samples: int = _DEFAULT_MAX_SAMPLES,
    ):
        super().__init__(max_samples)
        self.max_buffer_size = max_buffer_size

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        thresholds, n_pred = self._uniform_threshold_sampling(y_pred)
        rocs = np.zeros(self.max_buffer_size + 1)
        for buffer_size in range(self.max_buffer_size + 1):
            _, rocs[buffer_size] = self._range_pr_roc_auc_support(
                y_true, y_pred, buffer_size, thresholds, n_pred
            )
        return float(np.mean(rocs))
