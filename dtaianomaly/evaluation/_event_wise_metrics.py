import numba as nb
import numpy as np

from dtaianomaly.evaluation._BinaryMetric import BinaryMetric
from dtaianomaly.evaluation._FBetaMixin import FBetaMixin
from dtaianomaly.utils import make_intervals, np_any_axis0, np_any_axis1

__all__ = ["EventWiseFBeta", "EventWiseRecall", "EventWisePrecision"]


@nb.njit(fastmath=True, cache=True, parallel=True)
def _compute_event_wise_metrics(y_true: np.ndarray, y_pred: np.ndarray):

    # --- 1. Point-Wise Calculations ---
    fp = np.sum((~y_true) & y_pred)  # Using boolean operators for clarity
    tn = np.sum((~y_true) & (~y_pred))

    # --- 2. Identify Segments/Events ---
    gt_starts, gt_ends = make_intervals(y_true)
    pred_starts, pred_ends = make_intervals(y_pred)

    num_gt_events = gt_starts.shape[0]
    num_pred_events = pred_starts.shape[0]

    # Handle edge cases early
    if num_gt_events == 0:
        return 0.0, 1.0
    if num_pred_events == 0:
        return 0.0, 0.0

    # Build interval overlap matrix for efficient calculation
    # This avoids repeated overlap calculations
    overlap_matrix = np.zeros(shape=(num_gt_events, num_pred_events), dtype=nb.bool)

    for i in nb.prange(num_gt_events):
        for j in nb.prange(num_pred_events):

            # Calculate overlap
            start_overlap = max(gt_starts[i], pred_starts[j])
            end_overlap = min(gt_ends[i], pred_ends[j])

            # Simple overlap check
            overlap_matrix[i, j] = start_overlap <= end_overlap

    # Count true positives - each GT event detected at least once
    gt_detected = np_any_axis1(overlap_matrix)
    tpe = np.sum(gt_detected)

    # Count false positives - predicted events that don't overlap with any GT
    pred_is_fp = ~np_any_axis0(overlap_matrix)
    fpe = np.sum(pred_is_fp)

    # Calculate metrics
    recall_event = tpe / num_gt_events

    # Point-level False Alarm Rate
    num_actual_negatives = tn + fp
    far_pt = fp / num_actual_negatives if num_actual_negatives > 0 else 0.0

    # Event-wise precision
    precision_event_ratio = tpe / (tpe + fpe) if (tpe + fpe) > 0 else 0.0
    precision_event = precision_event_ratio * (1.0 - far_pt)
    precision_event = max(0.0, precision_event)  # Guard against negative values

    return precision_event, recall_event


class EventWisePrecision(BinaryMetric):
    """
    Computes the Event-Wise Precision score :cite:`el2024multivariate`.

    Precision measures how accurately the model identifies anomalies.
    For the Event-Wise Precision, the true and false positives are
    considered at the event-level:

    - :math:`TP_e`: the number of ground truth anomalous events that fully or partially
      overlap with a detected segment.
    - :math:`FP_e`: the number of detected segments that do not overlap with any ground
      truth anomalous event.

    The precision is corrected by the false-alarm rate (FAR) to avoid a model which predicts
    all observations as anomalous to have a high score. The FAR is computed on the point-level:

    - :math:`FP`: the number of detected anomalous **points** that are not actually anomalous.

    We then compute the Event-Wise Precision as (with :math:`N`: the total number of normal points):

    .. math::

       \\text{Event-Wise Precision} = \\frac{TP_e}{TP_e + FP_e} \\times (1 - \\frac{FP}{N})

    See Also
    --------
    EventWiseRecall: Compute the event-wise Recall score.
    EventWiseFBeta: Compute the event-wise :math:`F_\\beta` score.

    Examples
    --------
    >>> from dtaianomaly.evaluation import EventWisePrecision
    >>> metric = EventWisePrecision()
    >>> y_true = [0, 0, 0, 1, 1, 0, 0, 0]
    >>> y_pred = [1, 0, 0, 1, 1, 1, 0, 0]
    >>> metric.compute(y_true, y_pred)  # doctest: +ELLIPSIS
    0.333...
    """

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        event_wise_precision, _ = _compute_event_wise_metrics(
            y_true.astype(bool), y_pred.astype(bool)
        )
        return event_wise_precision


class EventWiseRecall(BinaryMetric):
    """
    Computes the Event-Wise Recall score :cite:`el2024multivariate`.

    Recall measures the model's ability to correctly identify all actual
    anomalies. For the Event-Wise Recall, the true positives and false
    negatives are considered at the event-level:

    - :math:`TP_e`: the number of ground truth anomalous events that fully or partially
      overlap with a detected segment.
    - :math:`FN_e`: the number of ground truth anomalous events that do not overlap with
      a detected segment.

    We then compute the Event-Wise Recall as:

    .. math::

       \\text{Event-Wise Recall} = \\frac{TP_e}{TP_e + FN_e}

    See Also
    --------
    EventWisePrecision: Compute the event-wise Precision score.
    EventWiseFBeta: Compute the event-wise :math:`F_\\beta` score.

    Examples
    --------
    >>> from dtaianomaly.evaluation import EventWiseRecall
    >>> metric = EventWiseRecall()
    >>> y_true = [0, 0, 0, 1, 1, 0, 0, 0]
    >>> y_pred = [1, 0, 0, 1, 1, 1, 0, 0]
    >>> metric.compute(y_true, y_pred)
    1.0
    """

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        _, event_wise_recall = _compute_event_wise_metrics(
            y_true.astype(bool), y_pred.astype(bool)
        )
        return event_wise_recall


class EventWiseFBeta(BinaryMetric, FBetaMixin):
    """
    Computes the Event-Wise :math:`F_\\beta` score :cite:`el2024multivariate`.

    The :math:`F_\\beta` combines both precision and recall into a single
    value. It provides a balanced evaluation of a model’s performance,
    especially in anomaly detection, where there is often a trade-off
    between catching all anomalies (high recall) and minimizing false
    alarms (high precision). The parameter :math:`\\beta` controls the balance
    between precision and recall. A :math:`\\beta > 1` gives more weight to
    recall, useful when missing anomalies is costly, while :math:`\\beta < 1`
    emphasizes precision, reducing false positives.

    The :math:`F_\\beta` score is the harmonic mean of the Event-Wise Precision
    and Event-Wise Recall.

    Parameters
    ----------
    beta: int, float, default=1
        Desired beta parameter.

    See Also
    --------
    EventWisePrecision: Compute the event-wise Precision score.
    EventWiseRecall: Compute the event-wise Recall score.

    Examples
    --------
    >>> from dtaianomaly.evaluation import EventWiseFBeta
    >>> metric = EventWiseFBeta()
    >>> y_true = [0, 0, 0, 1, 1, 0, 0, 0]
    >>> y_pred = [1, 0, 0, 1, 1, 1, 0, 0]
    >>> metric.compute(y_true, y_pred)
    0.5
    """

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        event_wise_precision, event_wise_recall = _compute_event_wise_metrics(
            y_true.astype(bool), y_pred.astype(bool)
        )
        return self._f_score(precision=event_wise_precision, recall=event_wise_recall)
