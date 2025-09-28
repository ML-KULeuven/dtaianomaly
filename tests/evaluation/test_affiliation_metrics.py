import math

import numpy as np
import pytest

from dtaianomaly.evaluation import (
    AffiliationFBeta,
    AffiliationPrecision,
    AffiliationRecall,
)
from dtaianomaly.evaluation._affiliation_metrics import (
    _affiliation_partition,
    _affiliation_precision_proba,
    _affiliation_recall_proba,
    _compute_affiliation_metrics,
    _cut_into_three_func,
    _cut_J_based_on_mean_func,
    _E_gt_func,
    _get_all_E_gt_func,
    _get_pivot_j,
    _integral_interval_distance,
    _integral_interval_probaCDF_precision,
    _integral_interval_probaCDF_recall,
    _integral_mini_interval,
    _integral_mini_interval_P_CDFmethod__min_piece,
    _integral_mini_interval_Pprecision_CDFmethod,
    _interval_intersection,
    _interval_length,
    _interval_subset,
    _len_wo_nan,
    _sum_interval_lengths,
    _sum_wo_nan,
    _test_events,
)

"""
Due to checks, two lines in the code are not reachable:
- "raise ValueError("The i_pivot should be outside J")" in _integral_mini_interval_Precall_CDFmethod
- "raise ValueError("unexpected unconsidered case")" in _cut_into_three_func
"""


class TestComputeAffiliationMetrics:

    @pytest.mark.parametrize(
        "y_true,y_pred,precision,recall",
        [
            (
                np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                math.nan,
                0,
            ),
            (
                np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 1]),
                np.array([0, 0, 0, 0, 1, 0, 0, 0, 1, 0]),
                0.8181818181818181,
                0.8442760942760943,
            ),
        ],
    )
    def test(self, y_true, y_pred, precision, recall):
        p, r = _compute_affiliation_metrics(y_true, y_pred)
        p_cls = AffiliationPrecision().compute(y_true, y_pred)
        r_cls = AffiliationRecall().compute(y_true, y_pred)

        if math.isnan(precision):
            assert math.isnan(p)
            assert math.isnan(p_cls)
        else:
            assert pytest.approx(p) == precision
            assert pytest.approx(p_cls) == precision
        assert pytest.approx(r) == recall
        assert pytest.approx(r_cls) == recall

    @pytest.mark.parametrize("beta", [0.5, 1, 2])
    def test_fbeta(self, beta):
        y_true = np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 1, 0, 0, 0, 1, 0])
        precision, recall = 0.8181818181818181, 0.8442760942760943
        numerator = (1 + beta**2) * precision * recall
        denominator = beta**2 * precision + recall
        assert pytest.approx(numerator / denominator) == AffiliationFBeta(beta).compute(
            y_true, y_pred
        )

    def test_no_ground_truth(self):
        with pytest.raises(ValueError):
            _compute_affiliation_metrics(np.zeros(shape=20), np.zeros(shape=20))


class TestAffiliationZone:

    def test_E_gt_func(self):
        Trange = (0, 30)
        events_gt = [(3, 7), (10, 18), (20, 21)]

        assert _E_gt_func(0, events_gt, Trange)[0] == 0
        assert _E_gt_func(0, events_gt, Trange)[1] == (10 + 7) / 2

        assert _E_gt_func(1, events_gt, Trange)[0] == (10 + 7) / 2
        assert _E_gt_func(1, events_gt, Trange)[1] == (18 + 20) / 2

        assert _E_gt_func(2, events_gt, Trange)[0] == (18 + 20) / 2
        assert _E_gt_func(2, events_gt, Trange)[1] == 30

    def test_E_gt_func_j_is_1(self):
        # Case j = 1
        Trange = (0, 30)
        events_gt = [(3, 20)]

        assert _E_gt_func(0, events_gt, Trange)[0] == 0
        assert _E_gt_func(0, events_gt, Trange)[1] == 30

    def test_get_all_E_gt_func(self):
        Trange = (0, 30)
        events_gt = [(3, 7), (10, 18), (20, 21)]
        cut_aff2 = _get_all_E_gt_func(events_gt, Trange)

        assert cut_aff2[0] == (0, (10 + 7) / 2)
        assert cut_aff2[1] == ((10 + 7) / 2, (18 + 20) / 2)
        assert cut_aff2[2] == ((18 + 20) / 2, 30)

    def test_affiliation_partition_precision_direction(self):
        """
        Test of the function in the 'precision' direction I --> J  in one example
        """
        events_gt = [(1, 8), (16, 17), (25, 28), (29, 31)]
        events_pred = [(1, 3), (6, 18), (25, 26)]
        M = _affiliation_partition(events_pred, _get_all_E_gt_func(events_gt, (0, 40)))

        # Check of dimension of the lists
        assert len(M) == len(events_gt)
        assert len(M[0]) == len(events_pred)

        # First element, related to the first affiliation zone (0, 12)
        assert M[0][0] == (1, 3)  # zone1, 1st prediction
        assert M[0][1] == (6, 12)  # zone1, 2nd prediction
        assert M[0][2] is None  # zone1, 3rd prediction

        # Second element, related to the second affiliation zone (12, 21)
        assert M[1][0] is None  # zone2, 1st prediction
        assert M[1][1] == (12, 18)  # zone2, 2nd prediction
        assert M[1][2] is None  # zone2, 3rd prediction

        # Third element, related to the third affiliation zone (25, 28)
        assert M[2][0] is None  # zone3, 1st prediction
        assert M[2][1] is None  # zone3, 2nd prediction
        assert M[2][2] == (25, 26)  # zone3, 3rd prediction

        # Fourth element, related to the fourth affiliation zone (29, 40)
        assert M[3][0] is None  # zone4, 1st prediction
        assert M[3][1] is None  # zone4, 2nd prediction
        assert M[3][2] is None  # zone4, 3rd prediction

    @pytest.mark.parametrize("nb_gt", [1, 2, 3])
    @pytest.mark.parametrize("nb_pred", [0, 1, 2, 3])
    def test_affiliation_partition_size(self, nb_gt, nb_pred):
        """
        Test of shape of the output of the function with only
        one prediction and one ground truth intervals
        """
        events_gt = [((i + 3) * 6, (i + 2) * 6 + 2) for i in range(nb_gt)]
        events_pred = [((i + 2) * 5, (i + 2) * 5 + 3) for i in range(nb_pred)]

        E_gt = _get_all_E_gt_func(
            events_gt, (0, max((nb_gt + 2) * 6 + 2, (nb_pred + 2) * 5 + 3) + 10)
        )
        M = _affiliation_partition(events_pred, E_gt)

        assert len(M) == nb_gt
        assert len(M[0]) == nb_pred


class TestIntegral:

    @pytest.mark.parametrize(
        "interval,expected",
        [
            (None, 0),
            ((1, 2), 1),
            ((-1, 3.5), 4.5),
        ],
    )
    def test_interval_length(self, interval, expected):
        assert _interval_length(interval) == expected

    @pytest.mark.parametrize(
        "intervals,expected",
        [
            ([], 0),
            ([(1, 2)], 1),
            ([(1, 2), (3.5, 4)], 1.5),
            ([(1, 2), (-1, 3.5)], 5.5),
        ],
    )
    def test_sum_interval_lengths(self, intervals, expected):
        assert _sum_interval_lengths(intervals) == expected

    @pytest.mark.parametrize(
        "interval1,interval2,expected",
        [
            (None, None, None),
            (None, (1, 2), None),
            ((1, 2), None, None),
            ((1, 2), (1, 2), (1, 2)),
            ((1, 2), (2, 3), None),
            ((1, 2), (3, 4), None),
            ((1, 3), (2, 4), (2, 3)),
            ((1, 3), (-1, 5), (1, 3)),
            ((1, 10), (0, 5), (1, 5)),
        ],
    )
    def test_interval_intersection(self, interval1, interval2, expected):
        assert _interval_intersection(interval1, interval2) == expected
        assert _interval_intersection(interval2, interval1) == expected  # Symmetric

    @pytest.mark.parametrize(
        "interval1,interval2",
        [
            (None, None),
            (None, (1, 2)),
            ((1, 2), None),
        ],
    )
    def test_interval_subset_invalid(self, interval1, interval2):
        with pytest.raises(TypeError):
            _interval_subset(interval1, interval2)

    @pytest.mark.parametrize(
        "interval1,interval2,expected",
        [
            ((1, 2), (1, 2), True),
            ((1, 2), (1, 3), True),
            ((1, 2), (0, 3), True),
            ((1, 3), (2, 3), False),
            ((1, 3), (-1, 2), False),
            ((1, 3), (-1, 0), False),
        ],
    )
    def test_interval_subset(self, interval1, interval2, expected):
        assert _interval_subset(interval1, interval2) == expected

    @pytest.mark.parametrize(
        "interval1,interval2,expected_cut1,expected_cut2,expected_cut3",
        [
            ((0, 1.5), (1, 2), (0, 1), (1, 1.5), None),
            ((-1, 10), (1.4, 2.4), (-1, 1.4), (1.4, 2.4), (2.4, 10)),
            ((-1, 1), (1.4, 2.4), (-1, 1), None, None),
            ((1.6, 2), (1.4, 2.4), None, (1.6, 2), None),
            ((4, 5), (1.4, 2.4), None, None, (4, 5)),
        ],
    )
    def test_cut_into_three(
        self, interval1, interval2, expected_cut1, expected_cut2, expected_cut3
    ):
        cuts = _cut_into_three_func(interval1, interval2)
        assert len(cuts) == 3
        assert cuts[0] == expected_cut1
        assert cuts[1] == expected_cut2
        assert cuts[2] == expected_cut3

    def test_get_pivot_j_error(self):
        with pytest.raises(ValueError):
            _get_pivot_j((0, 1.5), (1.4, 2.4))  # Overlap

    @pytest.mark.parametrize(
        "interval1,interval2,expected",
        [
            ((4, 5), (1.4, 2.4), 2.4),
            ((0, 1), (1.4, 2.4), 1.4),
        ],
    )
    def test_get_pivot_j(self, interval1, interval2, expected):
        assert _get_pivot_j(interval1, interval2) == expected

    def test_integral_mini_interval_error(self):
        with pytest.raises(ValueError):
            _integral_mini_interval((0, 1.5), (1.4, 2.4))  # None-empty intersection

    @pytest.mark.parametrize(
        "interval1,interval2,expected",
        [
            # We look at sum distance between every element of [4,5] to 2.4 the closest element of J
            # Distance is going from 4-2.4 to 5-2.4 i.e. 1.6 to 2.6, and increases linearly
            # There is 1.6 with a time duration of 1, and in addition the triangle 1/2 (integral from 0 to 1 of tdt)
            # Globally 1.6+1/2
            ((4, 5), (1.4, 2.4), 1.6 + 0.5),
            # We look at sum distance between every element of [0.1,1.2] to 1.4 the closest element of J
            # Distance is going from 1.3 to 0.2 and decreases linearly
            # There is 0.2 with a time duration of deltaI=1.1, and in addition
            # a decreases from 1.1 to 0 during 1.1 (integral from 0 to 1.1 of tdt) which is 1.1^2/2
            # Globally 0.2*1.1+1.1^2/2
            ((0.1, 1.2), (1.4, 2.4), 0.2 * 1.1 + 1.1**2 / 2),
        ],
    )
    def test_integral_mini_interval(self, interval1, interval2, expected):
        assert pytest.approx(_integral_mini_interval(interval1, interval2)) == expected

    @pytest.mark.parametrize(
        "interval1,interval2,expected",
        [
            ((0, 1.5), (-1, 2.4), 0),
            ((-1, 2.4), (-1, 2.4), 0),
            ((-10, 20), (-1, 2.4), 195.38),
            (
                (-10, 1.5),
                (-1, 2.4),
                _integral_interval_distance((-10, -1), (-1, 2.4)),
            ),  # The integral is same from I or I\J
        ],
    )
    def test_integral_interval(self, interval1, interval2, expected):
        assert (
            pytest.approx(_integral_interval_distance(interval1, interval2)) == expected
        )

    @pytest.mark.parametrize(
        "i_min,i_max,j_min,j_max,e_min,e_max,expected,case",
        [
            # Case 1: $d_max <= m$:
            # C = \int_{d_min}^{d_max} x dx = (1/2)*(d_max^2 - d_min^2)
            (0.924204, 1.376826, 1.570739, 1.903998, 0.7176185, 2.722883, 0.1902024, 1),
            # Case 2: $d_min < m < d_max$:
            # C = \int_{d_min}^{m} x dx + \int_{m}^{d_max} m dx
            #   = (1/2)*(m^2 - d_min^2) + m (d_max - m)
            (
                0.8751017,
                1.116294,
                0.5569796,
                0.8238064,
                0.3253522,
                1.403741,
                0.03960695,
                2,
            ),
            # Case 3: $m <= d_min$:
            # C = \int_{d_min}^{d_max} m dx = m (d_max - d_min)
            (
                0.767282,
                0.7753016,
                1.523338,
                1.958426,
                0.6516738,
                2.435003,
                0.003821954,
                3,
            ),
        ],
    )
    def test_integral_mini_interval_P_CDFmethod__min_piece(
        self, i_min, i_max, j_min, j_max, e_min, e_max, expected, case
    ):
        d_min = max(i_min - j_max, j_min - i_max)
        d_max = max(i_max - j_max, j_min - i_min)
        m = min(j_min - e_min, e_max - j_max)
        if case == 1:
            assert d_max <= m
        elif case == 2:
            assert d_min < m < d_max
        elif case == 3:
            assert m <= d_min

        A = min(d_max, m) ** 2 - min(d_min, m) ** 2  # 0.3804049
        B = max(d_max, m) - max(d_min, m)  # 0
        C = (1 / 2) * A + m * B  # 0.1902024
        assert pytest.approx(expected, rel=1e-5) == C

        # Actual test
        I = (i_min, i_max)
        J = (j_min, j_max)
        E = (e_min, e_max)
        assert (
            pytest.approx(
                _integral_mini_interval_P_CDFmethod__min_piece(I, J, E), rel=1e-5
            )
            == C
        )

    def test_integral_mini_interval_P_CDFmethod__min_piece_non_empty_intersection(self):
        with pytest.raises(ValueError):
            _integral_mini_interval_P_CDFmethod__min_piece((5, 10), (7.5, 20), (0, 30))

    def test_integral_mini_interval_P_CDFmethod__min_piece_I_not_in_E(self):
        with pytest.raises(ValueError):
            _integral_mini_interval_P_CDFmethod__min_piece((5, 10), (15, 20), (10, 30))

    def test_integral_mini_interval_P_CDFmethod__min_piece_J_not_in_E(self):
        with pytest.raises(ValueError):
            _integral_mini_interval_P_CDFmethod__min_piece((5, 10), (15, 20), (0, 12.5))

    @pytest.mark.parametrize(
        "j_min,j_max,e_min,e_max",
        [
            (0.9202326, 1.187741, 0.2655087, 1.842465),
            (0.7253212, 0.9439665, 0.3721239, 1.297164),
            (0.8431135, 1.35991, 0.5728534, 1.63017),
        ],
    )
    def test_integral_mini_interval_Pprecision_CDFmethod_symmetric(
        self, j_min, j_max, e_min, e_max
    ):
        # Explanation:
        # In case of symmetry the value is 1 for elements in J,
        # outside, it goes from (1 - DeltaJ/DeltaE) the closer to J,
        # until 0 at min(E) and max(E).
        # Since it's symmetric it decreases always linearly, on both side
        # It is (1 - DeltaJ/DeltaE) and not 1 as the border of J because
        # there is already DeltaJ/DeltaE of the probability took on the interval J
        #
        # So e.g. on the left, it's a triangle of height (1 - DeltaJ/DeltaE) and length
        # m (or M, it's the same since it's symmetric), so the answer.

        # on the left
        i_min_left = e_min
        i_max_left = j_min
        # on the right
        i_min_right = j_max
        i_max_right = e_max
        assert pytest.approx(min(j_min - e_min, e_max - j_max), rel=1e-5) == max(
            j_min - e_min, e_max - j_max
        )

        # Actual test
        I_left = (i_min_left, i_max_left)
        I_right = (i_min_right, i_max_right)
        J = (j_min, j_max)
        E = (e_min, e_max)
        integral_left = _integral_mini_interval_Pprecision_CDFmethod(I_left, J, E)
        integral_middle = max(J) - min(J)
        integral_right = _integral_mini_interval_Pprecision_CDFmethod(I_right, J, E)
        m = min(J) - min(E)
        M = max(E) - max(J)
        DeltaJ = max(J) - min(J)
        DeltaE = max(E) - min(E)

        assert pytest.approx((1 - DeltaJ / DeltaE) * m / 2, rel=1e-5) == integral_left
        assert pytest.approx(DeltaJ, rel=1e-5) == integral_middle
        assert pytest.approx((1 - DeltaJ / DeltaE) * M / 2, rel=1e-5) == integral_right

    @pytest.mark.parametrize(
        "j_min,e_min,e_max",
        [
            (0.9202326, 0.2655087, 1.842465),
            (0.7253212, 0.3721239, 1.297164),
            (0.8431135, 0.5728534, 1.63017),
        ],
    )
    def test_integral_mini_interval_Pprecision_CDFmethod_almost_point(
        self, j_min, e_min, e_max
    ):
        # Explanation: for point anomaly, the mean value should be 1/2
        j_max = j_min + 1e-9  # almost point case

        # on the left
        i_min_left = e_min
        i_max_left = j_min
        # on the right
        i_min_right = j_max
        i_max_right = e_max

        # Actual test
        I_left = (i_min_left, i_max_left)
        I_right = (i_min_right, i_max_right)
        J = (j_min, j_max)
        E = (e_min, e_max)
        integral_left = _integral_mini_interval_Pprecision_CDFmethod(I_left, J, E)
        # integral_middle = max(J) - min(J)
        integral_right = _integral_mini_interval_Pprecision_CDFmethod(I_right, J, E)
        DeltaE = max(E) - min(E)

        assert pytest.approx((integral_left + integral_right) / DeltaE) == 1 / 2

    @pytest.mark.parametrize(
        "E,J",
        [
            ((-3, 3), (-1, 2.4)),
        ],
    )
    def test_integral_interval_probaCDF_precision_basics(self, E, J):
        ## For I close to the border of E, it's close to 0%
        # (after taking the mean i.e. dividing by |I|)
        E = (-3, 3)
        J = (-1, 2.4)
        I1 = (E[0], -2.5)
        DeltaI1 = I1[1] - I1[0]
        I2 = (E[0], -2.8)
        DeltaI2 = I2[1] - I2[0]
        I3 = (E[0], -2.99)
        DeltaI3 = I3[1] - I3[0]

        assert _integral_interval_probaCDF_precision(I1, J, E) / DeltaI1 < 0.05
        assert (
            _integral_interval_probaCDF_precision(I2, J, E) / DeltaI2
            < _integral_interval_probaCDF_precision(I1, J, E) / DeltaI1
        )
        assert (
            _integral_interval_probaCDF_precision(I3, J, E) / DeltaI3
            < _integral_interval_probaCDF_precision(I2, J, E) / DeltaI2
        )

    @pytest.mark.parametrize(
        "E,J",
        [
            ((-3, 3), (-1, 2.4)),
            ((-10, 3), (0, 2.9)),
        ],
    )
    def test_integral_interval_probaCDF_precision_closed(self, E, J):
        # The total integral (when I is the whole interval E) is given by the sum:
        # I = (1-DeltaJ/DeltaE)*m/2 + (1-DeltaJ/DeltaE)*M/2 + DeltaJ
        # and M+m = DeltaE - DeltaJ so
        # I = (1-DeltaJ/DeltaE)*(DeltaE - DeltaJ)/2 + DeltaJ
        #   = (DeltaE - DeltaJ - DeltaJ + DeltaJ^2/DeltaE + 2*DeltaJ)/2         (*)
        #   = (DeltaE + DeltaJ^2/DeltaE)/2
        DeltaE = max(E) - min(E)
        DeltaJ = max(J) - min(J)
        closed_form = (DeltaE + DeltaJ**2 / DeltaE) / 2
        assert (
            pytest.approx(_integral_interval_probaCDF_precision(E, J, E)) == closed_form
        )

    @pytest.mark.parametrize(
        "interval,e_mean,expected",
        [
            (None, 1.5, (None, None)),
            ((2, 3), 1.5, (None, (2, 3))),
            ((0, 1), 1.5, ((0, 1), None)),
            ((0, 5), 1.5, ((0, 1.5), (1.5, 5))),
            ((0, 1.5), 1.5, ((0, 1.5), None)),
            ((1.5, 2), 1.5, (None, (1.5, 2))),
        ],
    )
    def test_cut_J_based_on_mean_func(self, interval, e_mean, expected):
        assert _cut_J_based_on_mean_func(interval, e_mean) == expected

    @pytest.mark.parametrize(
        "I,J,E,expected",
        [
            ### Almost point
            # I is at position J, so the recall should be 1
            ((2, 2), (2, 2), (1, 3), 1),
            # I is at middle between max(E) and min(J), so the recall should be 0.5
            ((1.5, 1.5), (2, 2), (1, 3), 0.5),
            ((2.5, 2.5), (2, 2), (1, 3), 0.5),
            # I is at the edge of E, the recall should be 0
            ((1, 1), (2, 2), (1, 3), 0),
            ((3, 3), (2, 2), (1, 3), 0),
            # I is outside E, the recall should be 0
            ((-4, -4), (2, 2), (1, 3), 0),
            ((10, 10), (2, 2), (1, 3), 0),
            ### Partially almost point
            # J is included in I, so the recall should be 1
            ((1, 3), (2, 2), (1, 3), 1),
            # I is at middle between max(E) and min(J), so the recall should be 0.5
            ((1, 1.5), (2, 2), (1, 3), 0.5),
            # Same for I at the other side
            ((2.5, 3), (2, 2), (1, 3), 0.5),
            # I is at the edge of E, the recall should be 0
            ((0, 1), (2, 2), (1, 3), 0),
            ((3, 5), (2, 2), (1, 3), 0),
            ### Special cases
            ((2, 2), (1, 3), (1, 3), 5 / 8),
            ((1, 1), (1, 3), (1, 3), 0.25),
            ((3, 3), (1, 3), (1, 3), 0.25),
        ],
    )
    def test_integral_interval_probaCDF_recall_almost_equal(
        self, I, J, E, expected, size_event=1e-9
    ):
        I = (I[0] - size_event, I[1] + size_event)
        J = (J[0] - size_event, J[1] + size_event)
        DeltaJ = max(J) - min(J)  # divide by J the size to obtain the mean
        assert (
            pytest.approx(
                _integral_interval_probaCDF_recall(I, J, E) / DeltaJ, abs=1e-5
            )
            == expected
        )

    @pytest.mark.parametrize(
        "I,J,E,expected",
        [
            # Almost point
            ((1.98, 1.98), (2, 2), (1, 3), 0.95),
            # Partially almost point
            ((1, 1.98), (2, 2), (1, 3), 0.95),
            # Increasing I
            ((0, 0), (-3, 3), (-5, 5), 0.625),
            ((-1, 1), (-3, 3), (-5, 5), 0.625),
            ((-2, 2), (-3, 3), (-5, 5), 0.625),
        ],
    )
    def test_integral_interval_probaCDF_recall_larger(
        self, I, J, E, expected, size_event=1e-9
    ):
        I = (I[0] - size_event, I[1] + size_event)
        J = (J[0] - size_event, J[1] + size_event)
        DeltaJ = max(J) - min(J)
        assert _integral_interval_probaCDF_recall(I, J, E) / DeltaJ > expected

    @pytest.mark.parametrize(
        "I,J,E,expected",
        [
            # Almost point
            ((1.01, 1.01), (2, 2), (1, 3), 0.1),
            ((2.99, 2.99), (2, 2), (1, 3), 0.1),
            # Partially almost point
            ((1, 1.01), (2, 2), (1, 3), 0.1),
            ((2.99, 3), (2, 2), (1, 3), 0.1),
        ],
    )
    def test_integral_interval_probaCDF_recall_smaller(
        self, I, J, E, expected, size_event=1e-9
    ):
        I = (I[0] - size_event, I[1] + size_event)
        J = (J[0] - size_event, J[1] + size_event)
        DeltaJ = max(J) - min(J)
        assert _integral_interval_probaCDF_recall(I, J, E) / DeltaJ < expected

    @pytest.mark.parametrize(
        "I1,I2,J,E",
        [
            ((-2, 2), (-1, 1), (-3, 3), (-5, 5)),
            ((-2.9, 2.9), (-2, 2), (-3, 3), (-5, 5)),
        ],
    )
    def test_integral_interval_probaCDF_recall_better_recall_larger_I(
        self, I1, I2, J, E, size_event=1e-9
    ):
        I1 = (I1[0] - size_event, I1[1] + size_event)
        I2 = (I2[0] - size_event, I2[1] + size_event)
        J = (J[0] - size_event, J[1] + size_event)
        DeltaJ = max(J) - min(J)
        assert (
            _integral_interval_probaCDF_recall(I1, J, E) / DeltaJ
            > _integral_interval_probaCDF_recall(I2, J, E) / DeltaJ
        )

    @pytest.mark.parametrize(
        "I,J,E1,E2",
        [
            ((10, 10), (-3, 3), (-10, 10), (-10, 12)),
            ((10, 10), (-3, 3), (-10, 12), (-10, 18)),
            ((10, 10), (-3, 3), (-10, 18), (-10, 30)),
            ((10, 10), (-3, 3), (-10, 30), (-10, 100)),
            ((10, 10), (-3, 3), (-10, 100), (-10, 1000)),
        ],
    )
    def test_integral_interval_probaCDF_recall_better_recall_larger_E(
        self, I, J, E1, E2, size_event=1e-9
    ):
        I = (I[0] - size_event, I[1] + size_event)
        J = (J[0] - size_event, J[1] + size_event)
        DeltaJ = max(J) - min(J)
        assert (
            _integral_interval_probaCDF_recall(I, J, E1) / DeltaJ
            < _integral_interval_probaCDF_recall(I, J, E2) / DeltaJ
        )


class TestSingleGroundTruthEvent:

    def test_affiliation_precision_proba_empty(self):
        assert math.isnan(_affiliation_precision_proba([], (1, 8), (1, 8)))
        assert math.isnan(_affiliation_precision_proba([None, None], (1, 8), (1, 8)))

    def test_affiliation_precision_proba_paper(self):
        J = (50 * 60, 70 * 60)
        Is = [(40 * 60, 60 * 60), (115 * 60, 120 * 60)]
        E = (30 * 60, 120 * 60)
        assert pytest.approx(_affiliation_precision_proba(Is, J, E)) == 0.672222222

    def test_affiliation_precision_proba_paper_with_none(self):
        J = (50 * 60, 70 * 60)
        Is = [(40 * 60, 60 * 60), (115 * 60, 120 * 60), None]
        E = (30 * 60, 120 * 60)
        assert pytest.approx(_affiliation_precision_proba(Is, J, E)) == 0.672222222

    def test_affiliation_recall_proba_empty(self):
        assert _affiliation_recall_proba([], (1, 8), (1, 8)) == 0
        assert _affiliation_recall_proba([None, None], (1, 8), (1, 8)) == 0

    def test_affiliation_recall_proba_paper(self):
        J = (50 * 60, 70 * 60)
        Is = [(40 * 60, 60 * 60), (115 * 60, 120 * 60)]
        E = (30 * 60, 120 * 60)
        assert pytest.approx(_affiliation_recall_proba(Is, J, E)) == 0.944444444

    def test_affiliation_recall_proba_paper_with_none(self):
        J = (50 * 60, 70 * 60)
        Is = [(40 * 60, 60 * 60), (115 * 60, 120 * 60), None]
        E = (30 * 60, 120 * 60)
        assert pytest.approx(_affiliation_recall_proba(Is, J, E)) == 0.944444444


class TestGenerics:

    @pytest.mark.parametrize(
        "vector,expected",
        [
            ([], 0),
            ([1, 4, 3], 8),
            ([1, math.nan, 3], 4),
            ([math.nan, math.nan, 3], 3),
            ([math.nan, math.nan, math.nan], 0),
        ],
    )
    def test_sum_wo_nan(self, vector, expected):
        assert _sum_wo_nan(vector) == expected

    @pytest.mark.parametrize(
        "vector,expected",
        [
            ([], 0),
            ([1, 4, 3], 3),
            ([1, math.nan, 3], 2),
            ([math.nan, math.nan, 3], 1),
            ([math.nan, math.nan, math.nan], 0),
        ],
    )
    def test_len_wo_nan(self, vector, expected):
        assert _len_wo_nan(vector) == expected


class TestTestEvents:

    @pytest.mark.parametrize(
        "events,error",
        [
            ((1, 3), TypeError),
            ([[1, 3], [4, 5]], TypeError),
            ([(1, 3), (4, 5, 6)], ValueError),
            ([(1, 3), (5, 4)], ValueError),
            ([(4, 6), (1, 2)], ValueError),
            ([(4, 6), (6, 7)], ValueError),
            ([(4, 5), (7, 7)], ValueError),
        ],
    )
    def test_invalid(self, events, error):
        with pytest.raises(error):
            _test_events(events)

    @pytest.mark.parametrize(
        "events",
        [
            [(1, 3)],
            [(1, 3), (5, 7)],
            [(1, 3), (5, 7), (100, 150)],
        ],
    )
    def test_valid(self, events):
        _test_events(events)
