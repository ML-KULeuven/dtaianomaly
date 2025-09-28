import math

import numpy as np

from dtaianomaly.evaluation._BinaryMetric import BinaryMetric
from dtaianomaly.evaluation._FBetaMixin import FBetaMixin
from dtaianomaly.utils import make_intervals

__all__ = [
    "AffiliationPrecision",
    "AffiliationRecall",
    "AffiliationFBeta",
]


###############################################################################
# AFFILIATION METRICS #########################################################
###############################################################################


def _test_events(events):
    """
    Verify the validity of the input events
    :param events: list of events, each represented by a couple (start, stop)
    :return: None. Raise an error for incorrect formed or non ordered events
    """
    if type(events) is not list:
        raise TypeError("Input `events` should be a list of couples")
    if not all([type(x) is tuple for x in events]):
        raise TypeError("Input `events` should be a list of tuples")
    if not all([len(x) == 2 for x in events]):
        raise ValueError("Input `events` should be a list of couples (start, stop)")
    if not all([x[0] < x[1] for x in events]):
        raise ValueError(
            "Input `events` should be a list of couples (start, stop) with start < stop"
        )
    if not all([events[i][1] < events[i + 1][0] for i in range(len(events) - 1)]):
        raise ValueError("Couples of input `events` should be disjoint and ordered")


def _compute_affiliation_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> (float, float):
    starts, ends = make_intervals(y_true)
    events_gt = list(zip(starts, ends + 1))
    _test_events(events_gt)

    starts, ends = make_intervals(y_pred)
    events_pred = list(zip(starts, ends + 1))
    _test_events(events_pred)

    if len(events_gt) == 0:
        raise ValueError("Input `events_gt` should have at least one event")

    E_gt = _get_all_E_gt_func(events_gt, (0, y_true.shape[0]))
    aff_partition = _affiliation_partition(events_pred, E_gt)

    # Computing precision
    p_precision = [
        _affiliation_precision_proba(Is, J, E)
        for Is, J, E in zip(aff_partition, events_gt, E_gt)
    ]

    # Computing recall
    p_recall = [
        _affiliation_recall_proba(Is, J, E)
        for Is, J, E in zip(aff_partition, events_gt, E_gt)
    ]

    if _len_wo_nan(p_precision) > 0:
        p_precision_average = _sum_wo_nan(p_precision) / _len_wo_nan(p_precision)
    else:
        p_precision_average = p_precision[0]  # math.nan
    p_recall_average = sum(p_recall) / len(p_recall)

    return float(p_precision_average), float(p_recall_average)


class AffiliationPrecision(BinaryMetric):
    """
    Compute the affiliation-based precision score :cite:`huet2022local`.

    The affiliation-metrics will first divide the time domain into a number
    of so-called affiliations: subsequences that are closest to the ground
    truth anomaly events. These affiliations do not have a fixed size. Then,
    the precision is computed within each affiliation as the distance from
    the predicted anomalous events to the ground truth event. The final
    precision then equals the average precision across all the affiliations.

    See Also
    --------
    AffiliationRecall: Compute the affiliation-based Recall score.
    AffiliationFBeta: Compute the affiliation-based :math:`F_\\beta` score.

    Examples
    --------
    >>> from dtaianomaly.evaluation import AffiliationPrecision
    >>> metric = AffiliationPrecision()
    >>> y_true = [0, 0, 0, 1, 1, 0, 0, 0]
    >>> y_pred = [1, 0, 0, 1, 1, 1, 0, 0]
    >>> metric.compute(y_true, y_pred)
    0.6875
    """

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        affiliation_precision, _ = _compute_affiliation_metrics(y_true, y_pred)
        return affiliation_precision


class AffiliationRecall(BinaryMetric):
    """
    Compute the affiliation-based recall score :cite:`huet2022local`.

    The affiliation-metrics will first divide the time domain into a number
    of so-called affiliations: subsequences that are closest to the ground
    truth anomaly events. These affiliations do not have a fixed size. Then,
    the recall is computed within each affiliation as the distance from
    the ground truth anomalous event to the closest predicted anomalies in
    that affiliation. The final recall then equals the average recall
    across all the affiliations.

    See Also
    --------
    AffiliationPrecision: Compute the affiliation-based Precision score.
    AffiliationFBeta: Compute the affiliation-based :math:`F_\\beta` score.

    Examples
    --------
    >>> from dtaianomaly.evaluation import AffiliationRecall
    >>> metric = AffiliationRecall()
    >>> y_true = [0, 0, 0, 1, 1, 0, 0, 0]
    >>> y_pred = [1, 0, 0, 1, 1, 1, 0, 0]
    >>> metric.compute(y_true, y_pred)
    1.0
    """

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        _, affiliation_recall = _compute_affiliation_metrics(y_true, y_pred)
        return affiliation_recall


class AffiliationFBeta(BinaryMetric, FBetaMixin):
    """
    Compute the affiliation-based :math:`F_\\beta` score :cite:`huet2022local`.

    The affiliation-metrics will first divide the time domain into a number
    of so-called affiliations: subsequences that are closest to the ground
    truth anomaly events. These affiliations do not have a fixed size. The
    precision is computed as the distance of the predicted anomalies to the
    ground truth event, and the recall is computed as the distance of the
    ground truth anomaly to the predicted anomalies. These precision and
    recall scores within each affiliation is then averaged. The :math:`F_\\beta`
    score is the harmonic mean of this average precision and recall.

    Parameters
    ----------
    beta : int, float, default=1
        Desired beta parameter.

    See Also
    --------
    AffiliationPrecision: Compute the affiliation-based Precision score.
    AffiliationRecall: Compute the affiliation-based Recall score.

    Examples
    --------
    >>> from dtaianomaly.evaluation import AffiliationFBeta
    >>> metric = AffiliationFBeta()
    >>> y_true = [0, 0, 0, 1, 1, 0, 0, 0]
    >>> y_pred = [1, 0, 0, 1, 1, 1, 0, 0]
    >>> metric.compute(y_true, y_pred)   # doctest: +ELLIPSIS
    0.814...
    """

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        affiliation_precision, affiliation_recall = _compute_affiliation_metrics(
            y_true, y_pred
        )
        return self._f_score(precision=affiliation_precision, recall=affiliation_recall)


###############################################################################
# AFFILIATION ZONES ###########################################################
###############################################################################


def _affiliation_partition(Is, E_gt):
    """
    Cut the events into the affiliation zones
    The presentation given here is from the ground truth point of view,
    but it is also used in the reversed direction in the main function.

    :param Is: events as a list of couples
    :param E_gt: range of the affiliation zones
    :return: a list of list of intervals (each interval represented by either
    a couple or None for empty interval). The outer list is indexed by each
    affiliation zone of `E_gt`. The inner list is indexed by the events of `Is`.
    """
    out = [None] * len(E_gt)
    for j in range(len(E_gt)):
        E_gt_j = E_gt[j]
        discarded_idx_before = [
            I[1] < E_gt_j[0] for I in Is
        ]  # end point of predicted I is before the begin of E
        discarded_idx_after = [
            I[0] > E_gt_j[1] for I in Is
        ]  # start of predicted I is after the end of E
        kept_index = [
            not (a or b) for a, b in zip(discarded_idx_before, discarded_idx_after)
        ]
        Is_j = [x for x, y in zip(Is, kept_index)]
        out[j] = [_interval_intersection(I, E_gt[j]) for I in Is_j]
    return out


def _t_start(j, Js, Trange):
    """
    Helper for `E_gt_func`

    :param j: index from 0 to len(Js) (included) on which to get the start
    :param Js: ground truth events, as a list of couples
    :param Trange: range of the series where Js is included
    :return: generalized start such that the middle of t_start and t_stop
    always gives the affiliation zone
    """
    b = max(Trange)
    n = len(Js)
    if j == n:
        return 2 * b - _t_stop(n - 1, Js, Trange)
    else:
        return Js[j][0]


def _t_stop(j, Js, Trange):
    """
    Helper for `E_gt_func`

    :param j: index from 0 to len(Js) (included) on which to get the stop
    :param Js: ground truth events, as a list of couples
    :param Trange: range of the series where Js is included
    :return: generalized stop such that the middle of t_start and t_stop
    always gives the affiliation zone
    """
    if j == -1:
        a = min(Trange)
        return 2 * a - _t_start(0, Js, Trange)
    else:
        return Js[j][1]


def _E_gt_func(j, Js, Trange):
    """
    Get the affiliation zone of element j of the ground truth

    :param j: index from 0 to len(Js) (excluded) on which to get the zone
    :param Js: ground truth events, as a list of couples
    :param Trange: range of the series where Js is included, can
    be (-math.inf, math.inf) for distance measures
    :return: affiliation zone of element j of the ground truth represented
    as a couple
    """
    range_left = (_t_stop(j - 1, Js, Trange) + _t_start(j, Js, Trange)) / 2
    range_right = (_t_stop(j, Js, Trange) + _t_start(j + 1, Js, Trange)) / 2
    return range_left, range_right


def _get_all_E_gt_func(Js, Trange):
    """
    Get the affiliation partition from the ground truth point of view

    :param Js: ground truth events, as a list of couples
    :param Trange: range of the series where Js is included, can
    be (-math.inf, math.inf) for distance measures
    :return: affiliation partition of the events
    """
    # E_gt is the limit of affiliation/attraction for each ground truth event
    return [_E_gt_func(j, Js, Trange) for j in range(len(Js))]


###############################################################################
# INTEGRAL INTERVAL #############################################################
###############################################################################


def _interval_length(J):
    """Length of an interval."""
    return 0 if J is None else J[1] - J[0]


def _sum_interval_lengths(Is):
    """Sum of length of the intervals."""
    return sum([_interval_length(I) for I in Is])


def _interval_intersection(I, J):
    """Intersection between two intervals I and J"""
    if I is None or J is None:
        return None

    I_inter_J = (max(I[0], J[0]), min(I[1], J[1]))
    if I_inter_J[0] >= I_inter_J[1]:
        return None
    else:
        return I_inter_J


def _interval_subset(I, J):
    """Checks whether I is a subset of J"""
    return (I[0] >= J[0]) and (I[1] <= J[1])


def _cut_into_three_func(I, J):
    """
    Cut an interval I into a partition of 3 subsets:
        the elements before J,
        the elements belonging to J,
        and the elements after J
    """
    if I is None:
        return None, None, None

    I_inter_J = _interval_intersection(I, J)
    if I == I_inter_J:
        I_before = None
        I_after = None
    elif I[1] <= J[0]:
        I_before = I
        I_after = None
    elif I[0] >= J[1]:
        I_before = None
        I_after = I
    elif (I[0] <= J[0]) and (I[1] >= J[1]):
        I_before = (I[0], I_inter_J[0])
        I_after = (I_inter_J[1], I[1])
    elif I[0] <= J[0]:
        I_before = (I[0], I_inter_J[0])
        I_after = None
    elif I[1] >= J[1]:
        I_before = None
        I_after = (I_inter_J[1], I[1])
    else:
        raise ValueError("unexpected unconsidered case")
    return I_before, I_inter_J, I_after


def _get_pivot_j(I, J):
    """
    Get the single point of J that is the closest to I, called 'pivot' here,
    with the requirement that I should be outside J
    """
    if max(I) <= min(J):
        return min(J)
    elif min(I) >= max(J):
        return max(J)
    else:
        raise ValueError("I and J should have a void intersection")


def _integral_mini_interval(I, J):
    """
    In the specific case where interval I is located outside J,
    integral of distance from x to J over the interval x \\in I.
    This is the *integral* i.e. the sum.
    It's not the mean (not divided by the length of I yet)
    """
    if I is None:
        return 0

    j_pivot = _get_pivot_j(I, J)
    a = min(I)
    b = max(I)

    return (b - a) * abs((j_pivot - (a + b) / 2))


def _integral_interval_distance(I, J):
    """
    For any non-empty intervals I, J, compute the
    integral of distance from x to J over the interval x \\in I.
    This is the *integral* i.e. the sum.
    It's not the mean (not divided by the length of I yet)
    The interval I can intersect J or not
    """

    # I and J are single intervals (not generic sets)
    # I is a predicted interval in the range of affiliation of J
    def f(I_cut):
        return _integral_mini_interval(I_cut, J)

    # If I_middle is fully included into J, it is the distance to J is always 0
    def f0(_):
        return 0

    cut_into_three = _cut_into_three_func(I, J)
    # Distance for now, not the mean:
    # Distance left: Between cut_into_three[0] and the point min(J)
    d_left = f(cut_into_three[0])
    # Distance middle: Between cut_into_three[1] = I inter J, and J
    d_middle = f0(cut_into_three[1])
    # Distance right: Between cut_into_three[2] and the point max(J)
    d_right = f(cut_into_three[2])
    # It's an integral so summable
    return d_left + d_middle + d_right


def _integral_mini_interval_P_CDFmethod__min_piece(I, J, E):
    """
    Helper of `integral_mini_interval_Pprecision_CDFmethod`
    In the specific case where interval I is located outside J,
    compute the integral $\\int_{d_min}^{d_max} \\min(m, x) dx$, with:
    - m the smallest distance from J to E,
    - d_min the smallest distance d(x, J) from x \\in I to J
    - d_max the largest distance d(x, J) from x \\in I to J
    """
    if _interval_intersection(I, J) is not None:
        raise ValueError("I and J should have a void intersection")
    if not _interval_subset(J, E):
        raise ValueError("J should be included in E")
    if not _interval_subset(I, E):
        raise ValueError("I should be included in E")

    e_min = min(E)
    j_min = min(J)
    j_max = max(J)
    e_max = max(E)
    i_min = min(I)
    i_max = max(I)

    d_min = max(i_min - j_max, j_min - i_max)
    d_max = max(i_max - j_max, j_min - i_min)
    m = min(j_min - e_min, e_max - j_max)
    A = min(d_max, m) ** 2 - min(d_min, m) ** 2
    B = max(d_max, m) - max(d_min, m)
    C = (1 / 2) * A + m * B

    return C


def _integral_mini_interval_Pprecision_CDFmethod(I, J, E):
    """
    Integral of the probability of distances over the interval I.
    In the specific case where interval I is located outside J,
    compute the integral $\\int_{x \\in I} Fbar(dist(x,J)) dx$.
    This is the *integral* i.e. the sum (not the mean)
    """
    integral_min_piece = _integral_mini_interval_P_CDFmethod__min_piece(I, J, E)

    e_min = min(E)
    j_min = min(J)
    j_max = max(J)
    e_max = max(E)
    i_min = min(I)
    i_max = max(I)
    d_min = max(i_min - j_max, j_min - i_max)
    d_max = max(i_max - j_max, j_min - i_min)
    integral_linear_piece = (1 / 2) * (d_max**2 - d_min**2)
    integral_remaining_piece = (j_max - j_min) * (i_max - i_min)

    DeltaI = i_max - i_min
    DeltaE = e_max - e_min

    return DeltaI - (1 / DeltaE) * (
        integral_min_piece + integral_linear_piece + integral_remaining_piece
    )


def _integral_interval_probaCDF_precision(I, J, E):
    """
    Integral of the probability of distances over the interval I.
    Compute the integral $\\int_{x \\in I} Fbar(dist(x,J)) dx$.
    This is the *integral* i.e. the sum (not the mean)
    """

    # I and J are single intervals (not generic sets)
    def f(I_cut):
        return (
            0
            if I_cut is None
            else _integral_mini_interval_Pprecision_CDFmethod(I_cut, J, E)
        )

    # If I_middle is fully included into J, it is
    # integral of 1 on the interval I_middle, so it's |I_middle|
    def f0(I_middle):
        return 0 if I_middle is None else max(I_middle) - min(I_middle)

    cut_into_three = _cut_into_three_func(I, J)
    # Distance for now, not the mean:
    # Distance left: Between cut_into_three[0] and the point min(J)
    d_left = f(cut_into_three[0])
    # Distance middle: Between cut_into_three[1] = I inter J, and J
    d_middle = f0(cut_into_three[1])
    # Distance right: Between cut_into_three[2] and the point max(J)
    d_right = f(cut_into_three[2])
    # It's an integral so summable
    return d_left + d_middle + d_right


def _cut_J_based_on_mean_func(J, e_mean):
    """
    Helper function for the recall.
    Partition J into two intervals: before and after e_mean
    (e_mean represents the center element of E the zone of affiliation)
    """
    if J is None:
        J_before = None
        J_after = None
    elif e_mean >= max(J):
        J_before = J
        J_after = None
    elif e_mean <= min(J):
        J_before = None
        J_after = J
    else:  # e_mean is across J
        J_before = (min(J), e_mean)
        J_after = (e_mean, max(J))

    return J_before, J_after


def _integral_mini_interval_Precall_CDFmethod(I, J, E):
    """
    Integral of the probability of distances over the interval J.
    In the specific case where interval J is located outside I,
    compute the integral $\\int_{y \\in J} Fbar_y(dist(y,I)) dy$.
    This is the *integral* i.e. the sum (not the mean)
    """
    # The interval J should be located outside I
    # (so it's either the left piece or the right piece w.r.t I)
    i_pivot = _get_pivot_j(J, I)
    e_min = min(E)
    e_max = max(E)
    e_mean = (e_min + e_max) / 2

    # If i_pivot is outside E (it's possible), then
    # the distance is worst that any random element within E,
    # so we set the recall to 0
    if i_pivot <= min(E):
        return 0
    elif i_pivot >= max(E):
        return 0
    # Otherwise, we have at least i_pivot in E and so d < M so min(d,M)=d

    cut_J_based_on_e_mean = _cut_J_based_on_mean_func(J, e_mean)
    J_before = cut_J_based_on_e_mean[0]
    J_after = cut_J_based_on_e_mean[1]

    iemin_mean = (e_min + i_pivot) / 2
    cut_Jbefore_based_on_iemin_mean = _cut_J_based_on_mean_func(J_before, iemin_mean)
    J_before_closeE = cut_Jbefore_based_on_iemin_mean[
        0
    ]  # before e_mean and closer to e_min than i_pivot ~ J_before_before
    J_before_closeI = cut_Jbefore_based_on_iemin_mean[
        1
    ]  # before e_mean and closer to i_pivot than e_min ~ J_before_after

    iemax_mean = (e_max + i_pivot) / 2
    cut_Jafter_based_on_iemax_mean = _cut_J_based_on_mean_func(J_after, iemax_mean)
    J_after_closeI = cut_Jafter_based_on_iemax_mean[
        0
    ]  # after e_mean and closer to i_pivot than e_max ~ J_after_before
    J_after_closeE = cut_Jafter_based_on_iemax_mean[
        1
    ]  # after e_mean and closer to e_max than i_pivot ~ J_after_after

    if J_before_closeE is not None:
        j_before_before_min = min(J_before_closeE)  # == min(J)
        j_before_before_max = max(J_before_closeE)
    else:
        j_before_before_min = math.nan
        j_before_before_max = math.nan

    if J_before_closeI is not None:
        j_before_after_min = min(J_before_closeI)  # == j_before_before_max if existing
        j_before_after_max = max(J_before_closeI)  # == max(J_before)
    else:
        j_before_after_min = math.nan
        j_before_after_max = math.nan

    if J_after_closeI is not None:
        j_after_before_min = min(J_after_closeI)  # == min(J_after)
        j_after_before_max = max(J_after_closeI)
    else:
        j_after_before_min = math.nan
        j_after_before_max = math.nan

    if J_after_closeE is not None:
        j_after_after_min = min(J_after_closeE)  # == j_after_before_max if existing
        j_after_after_max = max(J_after_closeE)  # == max(J)
    else:
        j_after_after_min = math.nan
        j_after_after_max = math.nan

    # <-- J_before_closeE --> <-- J_before_closeI --> <-- J_after_closeI --> <-- J_after_closeE -->
    # j_bb_min       j_bb_max j_ba_min       j_ba_max j_ab_min      j_ab_max j_aa_min      j_aa_max
    # (with `b` for before and `a` for after in the previous variable names)

    #                                          vs e_mean  m = min(t-e_min, e_max-t)  d=|i_pivot-t|   min(d,m)                            \int min(d,m)dt   \int d dt        \int_(min(d,m)+d)dt                                    \int_{t \in J}(min(d,m)+d)dt
    # Case J_before_closeE & i_pivot after J   before     t-e_min                    i_pivot-t       min(i_pivot-t,t-e_min) = t-e_min    t^2/2-e_min*t     i_pivot*t-t^2/2  t^2/2-e_min*t+i_pivot*t-t^2/2 = (i_pivot-e_min)*t      (i_pivot-e_min)*tB - (i_pivot-e_min)*tA = (i_pivot-e_min)*(tB-tA)
    # Case J_before_closeI & i_pivot after J   before     t-e_min                    i_pivot-t       min(i_pivot-t,t-e_min) = i_pivot-t  i_pivot*t-t^2/2   i_pivot*t-t^2/2  i_pivot*t-t^2/2+i_pivot*t-t^2/2 = 2*i_pivot*t-t^2      2*i_pivot*tB-tB^2 - 2*i_pivot*tA + tA^2 = 2*i_pivot*(tB-tA) - (tB^2 - tA^2)
    # Case J_after_closeI & i_pivot after J    after      e_max-t                    i_pivot-t       min(i_pivot-t,e_max-t) = i_pivot-t  i_pivot*t-t^2/2   i_pivot*t-t^2/2  i_pivot*t-t^2/2+i_pivot*t-t^2/2 = 2*i_pivot*t-t^2      2*i_pivot*tB-tB^2 - 2*i_pivot*tA + tA^2 = 2*i_pivot*(tB-tA) - (tB^2 - tA^2)
    # Case J_after_closeE & i_pivot after J    after      e_max-t                    i_pivot-t       min(i_pivot-t,e_max-t) = e_max-t    e_max*t-t^2/2     i_pivot*t-t^2/2  e_max*t-t^2/2+i_pivot*t-t^2/2 = (e_max+i_pivot)*t-t^2  (e_max+i_pivot)*tB-tB^2 - (e_max+i_pivot)*tA + tA^2 = (e_max+i_pivot)*(tB-tA) - (tB^2 - tA^2)
    #
    # Case J_before_closeE & i_pivot before J  before     t-e_min                    t-i_pivot       min(t-i_pivot,t-e_min) = t-e_min    t^2/2-e_min*t     t^2/2-i_pivot*t  t^2/2-e_min*t+t^2/2-i_pivot*t = t^2-(e_min+i_pivot)*t  tB^2-(e_min+i_pivot)*tB - tA^2 + (e_min+i_pivot)*tA = (tB^2 - tA^2) - (e_min+i_pivot)*(tB-tA)
    # Case J_before_closeI & i_pivot before J  before     t-e_min                    t-i_pivot       min(t-i_pivot,t-e_min) = t-i_pivot  t^2/2-i_pivot*t   t^2/2-i_pivot*t  t^2/2-i_pivot*t+t^2/2-i_pivot*t = t^2-2*i_pivot*t      tB^2-2*i_pivot*tB - tA^2 + 2*i_pivot*tA = (tB^2 - tA^2) - 2*i_pivot*(tB-tA)
    # Case J_after_closeI & i_pivot before J   after      e_max-t                    t-i_pivot       min(t-i_pivot,e_max-t) = t-i_pivot  t^2/2-i_pivot*t   t^2/2-i_pivot*t  t^2/2-i_pivot*t+t^2/2-i_pivot*t = t^2-2*i_pivot*t      tB^2-2*i_pivot*tB - tA^2 + 2*i_pivot*tA = (tB^2 - tA^2) - 2*i_pivot*(tB-tA)
    # Case J_after_closeE & i_pivot before J   after      e_max-t                    t-i_pivot       min(t-i_pivot,e_max-t) = e_max-t    e_max*t-t^2/2     t^2/2-i_pivot*t  e_max*t-t^2/2+t^2/2-i_pivot*t = (e_max-i_pivot)*t      (e_max-i_pivot)*tB - (e_max-i_pivot)*tA = (e_max-i_pivot)*(tB-tA)

    if i_pivot >= max(J):
        part1_before_closeE = (i_pivot - e_min) * (
            j_before_before_max - j_before_before_min
        )  # (i_pivot-e_min)*(tB-tA) # j_before_before_max - j_before_before_min
        part2_before_closeI = 2 * i_pivot * (
            j_before_after_max - j_before_after_min
        ) - (
            j_before_after_max**2 - j_before_after_min**2
        )  # 2*i_pivot*(tB-tA) - (tB^2 - tA^2) # j_before_after_max - j_before_after_min
        part3_after_closeI = 2 * i_pivot * (j_after_before_max - j_after_before_min) - (
            j_after_before_max**2 - j_after_before_min**2
        )  # 2*i_pivot*(tB-tA) - (tB^2 - tA^2) # j_after_before_max - j_after_before_min
        part4_after_closeE = (e_max + i_pivot) * (
            j_after_after_max - j_after_after_min
        ) - (
            j_after_after_max**2 - j_after_after_min**2
        )  # (e_max+i_pivot)*(tB-tA) - (tB^2 - tA^2) # j_after_after_max - j_after_after_min
        out_parts = [
            part1_before_closeE,
            part2_before_closeI,
            part3_after_closeI,
            part4_after_closeE,
        ]
    elif i_pivot <= min(J):
        part1_before_closeE = (j_before_before_max**2 - j_before_before_min**2) - (
            e_min + i_pivot
        ) * (
            j_before_before_max - j_before_before_min
        )  # (tB^2 - tA^2) - (e_min+i_pivot)*(tB-tA) # j_before_before_max - j_before_before_min
        part2_before_closeI = (
            j_before_after_max**2 - j_before_after_min**2
        ) - 2 * i_pivot * (
            j_before_after_max - j_before_after_min
        )  # (tB^2 - tA^2) - 2*i_pivot*(tB-tA) # j_before_after_max - j_before_after_min
        part3_after_closeI = (
            j_after_before_max**2 - j_after_before_min**2
        ) - 2 * i_pivot * (
            j_after_before_max - j_after_before_min
        )  # (tB^2 - tA^2) - 2*i_pivot*(tB-tA) # j_after_before_max - j_after_before_min
        part4_after_closeE = (e_max - i_pivot) * (
            j_after_after_max - j_after_after_min
        )  # (e_max-i_pivot)*(tB-tA) # j_after_after_max - j_after_after_min
        out_parts = [
            part1_before_closeE,
            part2_before_closeI,
            part3_after_closeI,
            part4_after_closeE,
        ]
    else:
        raise ValueError("The i_pivot should be outside J")

    out_integral_min_dm_plus_d = _sum_wo_nan(
        out_parts
    )  # integral on all J, i.e. sum of the disjoint parts

    # We have for each point t of J:
    # \bar{F}_{t, recall}(d) = 1 - (1/|E|) * (min(d,m) + d)
    # Since t is a single-point here, and we are in the case where i_pivot is inside E.
    # The integral is then given by:
    # C = \int_{t \in J} \bar{F}_{t, recall}(D(t)) dt
    #   = \int_{t \in J} 1 - (1/|E|) * (min(d,m) + d) dt
    #   = |J| - (1/|E|) * [\int_{t \in J} (min(d,m) + d) dt]
    #   = |J| - (1/|E|) * out_integral_min_dm_plus_d
    DeltaJ = max(J) - min(J)
    DeltaE = max(E) - min(E)
    C = DeltaJ - (1 / DeltaE) * out_integral_min_dm_plus_d

    return C


def _integral_interval_probaCDF_recall(I, J, E):
    """
    Integral of the probability of distances over the interval J.
    Compute the integral $\\int_{y \\in J} Fbar_y(dist(y,I)) dy$.
    This is the *integral* i.e. the sum (not the mean)
    """

    # I and J are single intervals (not generic sets)
    # E is the outside affiliation interval of J (even for recall!)
    # (in particular J \subset E)
    #
    # J is the portion of the ground truth affiliated to I
    # I is a predicted interval (can be outside E possibly since it's recall)
    def f(J_cut):
        return (
            0
            if J_cut is None
            else _integral_mini_interval_Precall_CDFmethod(I, J_cut, E)
        )

    # If J_middle is fully included into I, it is
    # integral of 1 on the interval J_middle, so it's |J_middle|
    def f0(J_middle):
        return 0 if J_middle is None else max(J_middle) - min(J_middle)

    cut_into_three = _cut_into_three_func(
        J, I
    )  # it's J that we cut into 3, depending on the position w.r.t I
    # since we integrate over J this time.

    # Distance for now, not the mean:
    # Distance left: Between cut_into_three[0] and the point min(I)
    d_left = f(cut_into_three[0])
    # Distance middle: Between cut_into_three[1] = J inter I, and I
    d_middle = f0(cut_into_three[1])
    # Distance right: Between cut_into_three[2] and the point max(I)
    d_right = f(cut_into_three[2])
    # It's an integral so summable
    return d_left + d_middle + d_right


###############################################################################
# SINGLE GROUND TRUTH EVENTS ##################################################
###############################################################################


def _affiliation_precision_proba(Is, J, E):
    """Compute the individual precision probability from Is to a single ground truth J"""
    if all([I is None for I in Is]):  # no prediction in the current area
        return math.nan  # undefined
    return sum(
        [_integral_interval_probaCDF_precision(I, J, E) for I in Is]
    ) / _sum_interval_lengths(Is)


def _affiliation_recall_proba(Is, J, E):
    """Compute the individual recall probability from a single ground truth J to Is"""
    Is = [I for I in Is if I is not None]  # filter possible None in Is
    if len(Is) == 0:  # there is no prediction in the current area
        return 0
    E_gt_recall = _get_all_E_gt_func(
        Is, E
    )  # here from the point of view of the predictions
    Js = _affiliation_partition(
        [J], E_gt_recall
    )  # partition of J depending of proximity with Is
    return sum(
        [_integral_interval_probaCDF_recall(I, J[0], E) for I, J in zip(Is, Js)]
    ) / _interval_length(J)


###############################################################################
# GENERICS ####################################################################
###############################################################################


def _sum_wo_nan(vec):
    """Sum of elements, ignoring math.isnan ones."""
    return sum([e for e in vec if not math.isnan(e)])


def _len_wo_nan(vec):
    """Count of elements, ignoring math.isnan ones."""
    return len([e for e in vec if not math.isnan(e)])
