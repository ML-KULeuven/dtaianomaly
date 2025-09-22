import numpy as np
import pytest

from dtaianomaly.evaluation.range_based_metrics import (
    RangeBasedFBeta,
    RangeBasedPrecision,
    RangeBasedRecall,
    _cardinality_factor,
    _delta,
    _existence_reward,
    _gamma,
    _interval_overlap,
    _omega,
    _overlap_reward,
    _precision_interval,
    _recall_interval,
)


@pytest.fixture
def test_instance():
    y_true = np.array(
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]
    )  # (5, 8), (11, 13), (19, 20)
    y_pred = np.array(
        [0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    )  # (2, 4), (6, 9), (11, 13)
    """
    Precision:
    - First predicted: 0
    - Second predicted: 2/3 = 4/6
    - third predicted: 2/2  =
    => precision = (0 + 2/3 + 3/3) / 3 = (5/3) / 3 = 5/9
    """
    precision = 5 / 9
    """"
    Recall (includes existence reward):
    - First true: 2/3 * 0.5 + 0.5  = 2/6 + 1/2 = 5/6
    - Second true: 2/2 * 0.5 + 0.5 = 2/4 + 1/2 = 4/4 = 6/6
    - third true: 0
    => recall = (5/6 + 6/6 + 0) / 3 = (11/6) / 3 = 11/18
    """
    recall = 11 / 18
    return y_true, y_pred, precision, recall


@pytest.mark.parametrize(
    "metric",
    [
        RangeBasedPrecision,
        RangeBasedRecall,
        RangeBasedFBeta,
    ],
)
class TestInitialization:

    def test_no_parameters(self, metric):
        metric()

    def test_alpha_invalid_type(self, metric):
        if isinstance(metric(), RangeBasedPrecision):
            return
        with pytest.raises(TypeError):
            metric(alpha="0.5")
        with pytest.raises(TypeError):
            metric(alpha=True)
        metric(alpha=0)
        metric(alpha=0.2)
        metric(alpha=0.9)

    def test_alpha_invalid_value(self, metric):
        if isinstance(metric(), RangeBasedPrecision):
            return
        with pytest.raises(ValueError):
            metric(alpha=-1)
        with pytest.raises(ValueError):
            metric(alpha=1.0001)
        metric(alpha=0)
        metric(alpha=0.5)
        metric(alpha=1)

    def test_delta_invalid_type(self, metric):
        with pytest.raises(TypeError):
            metric(delta=1)
        with pytest.raises(TypeError):
            metric(delta=0.5)
        with pytest.raises(TypeError):
            metric(delta=True)
        with pytest.raises(TypeError):
            metric(delta=lambda _: 1)
        with pytest.raises(TypeError):
            metric(delta=lambda _, __, ___: 1)
        metric(delta="flat")
        metric(delta="front")
        metric(delta="back")
        metric(delta="middle")
        metric(delta=lambda _, __: 1)

    def test_delta_invalid_value(self, metric):
        with pytest.raises(ValueError):
            metric(delta="1")
        metric(delta="flat")
        metric(delta="front")
        metric(delta="back")
        metric(delta="middle")
        metric(delta=lambda _, __: 1)

    def test_gamma_invalid_type(self, metric):
        with pytest.raises(TypeError):
            metric(gamma=1)
        with pytest.raises(TypeError):
            metric(gamma=0.5)
        with pytest.raises(TypeError):
            metric(gamma=True)
        with pytest.raises(TypeError):
            metric(gamma=lambda _, __: 1)
        with pytest.raises(TypeError):
            metric(gamma=lambda _, __, ___: 1)
        metric(gamma="one")
        metric(gamma="reciprocal")
        metric(gamma=lambda _: 1)

    def test_gamma_invalid_value(self, metric):
        with pytest.raises(ValueError):
            metric(gamma="1")
        metric(gamma="one")
        metric(gamma="reciprocal")
        metric(gamma=lambda _: 1)


class TestRangeBasedPrecision:

    def test(self, test_instance):
        y_true, y_pred, precision, _ = test_instance
        assert pytest.approx(RangeBasedPrecision().compute(y_true, y_pred)) == precision

    def test_str(self):
        assert str(RangeBasedPrecision()) == "RangeBasedPrecision()"
        assert (
            str(RangeBasedPrecision(delta="front"))
            == "RangeBasedPrecision(delta='front')"
        )


class TestRangeBasedRecall:

    def test(self, test_instance):
        y_true, y_pred, _, recall = test_instance
        assert pytest.approx(RangeBasedRecall().compute(y_true, y_pred)) == recall

    def test_str(self):
        assert str(RangeBasedRecall()) == "RangeBasedRecall()"
        assert str(RangeBasedRecall(alpha=0.3)) == "RangeBasedRecall(alpha=0.3)"
        assert str(RangeBasedRecall(delta="front")) == "RangeBasedRecall(delta='front')"


class TestRangeBasedFBeta:

    def test_default_beta(self):
        assert RangeBasedFBeta().beta == 1.0

    def test_string_beta(self):
        with pytest.raises(TypeError):
            RangeBasedFBeta("1.0")

    def test_bool_beta(self):
        with pytest.raises(TypeError):
            RangeBasedFBeta(True)

    def test_zero_beta(self):
        with pytest.raises(ValueError):
            RangeBasedFBeta(0.0)

    def test_negative_beta(self):
        with pytest.raises(ValueError):
            RangeBasedFBeta(-1.0)

    @pytest.mark.parametrize("beta", [0.5, 1, 2])
    def test(self, beta, test_instance):
        y_true, y_pred, precision, recall = test_instance
        numerator = (1 + beta**2) * precision * recall
        denominator = beta**2 * precision + recall
        assert pytest.approx(numerator / denominator) == RangeBasedFBeta(beta).compute(
            y_true, y_pred
        )

    def test_str(self):
        assert str(RangeBasedFBeta()) == "RangeBasedFBeta()"
        assert str(RangeBasedFBeta(beta=0.5)) == "RangeBasedFBeta(beta=0.5)"
        assert str(RangeBasedFBeta(beta=2)) == "RangeBasedFBeta(beta=2)"
        assert str(RangeBasedFBeta(alpha=0.3)) == "RangeBasedFBeta(alpha=0.3)"
        assert str(RangeBasedFBeta(delta="front")) == "RangeBasedFBeta(delta='front')"


class TestUtils:

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            ((1, 3), (5, 6), None),
            ((1, 3), (2, 6), (2, 3)),
            ((0, 5), (2, 3), (2, 3)),
        ],
    )
    def test_interval_overlap(self, a, b, expected):
        assert _interval_overlap(a, b) == expected

    @pytest.mark.parametrize(
        "anomaly_range,overlap_set,delta,expected",
        [
            # No overlap
            ((1, 10), None, "flat", 0),
            # Overlap
            ((1, 10), (5, 10), "flat", 5 / 9),
            ((1, 10), (5, 7), "flat", 2 / 9),
            ((5, 7), (5, 7), "flat", 2 / 2),
            # Front bias: 9, 8, 7, 6, 5, 4, 3, 2, 1
            ((1, 10), (1, 5), "front", 30 / 45),
            ((1, 10), (3, 6), "front", 18 / 45),
            ((1, 10), (5, 10), "front", 15 / 45),
            # Back bias: 1, 2, 3, 4, 5, 6, 7, 8, 9
            ((1, 10), (1, 5), "back", 10 / 45),
            ((1, 10), (3, 6), "back", 12 / 45),
            ((1, 10), (5, 10), "back", 35 / 45),
            # Middle bias: 1, 2, 3, 4, 5, 4, 3, 2, 1
            ((1, 10), (1, 5), "middle", 10 / 25),
            ((1, 10), (3, 7), "middle", 16 / 25),
            ((1, 10), (5, 10), "middle", 15 / 25),
        ],
    )
    def test_omega(self, anomaly_range, overlap_set, delta, expected):
        assert pytest.approx(_omega(anomaly_range, overlap_set, delta)) == expected

    def test_delta_flat(self):
        for i in range(20):
            assert _delta("flat", i, 20) == 1

    def test_delta_front(self):
        for i in range(20):
            assert _delta("front", i, 20) == 21 - i

    def test_delta_back(self):
        for i in range(20):
            assert _delta("back", i, 20) == i

    def test_delta_middle(self):
        for i in range(11):
            assert _delta("middle", i, 20) == i
        for i in range(11, 20):
            assert _delta("middle", i, 20) == 21 - i

    def test_delta_custom(self):
        for i in range(20):
            assert _delta(lambda j, _: j + 1, i, 20) == i + 1
        for i in range(20):
            assert _delta(lambda _, j: j, i, 20) == 20

    def test_gamma_one(self):
        for i in range(2, 10):
            assert _gamma("one", i) == 1

    def test_gamma_reciprocal(self):
        for i in range(2, 10):
            assert _gamma("reciprocal", i) == 1 / i

    def test_gamma_custom(self):
        for i in range(2, 20):
            assert _gamma(lambda j: 1 / (2 * j), i) == 1 / (2 * i)

    @pytest.mark.parametrize(
        "interval,other_intervals,expected",
        [
            # No existence
            ((1, 3), [(5, 6), (9, 20), (26, 28), (35, 40)], 0),
            ((7, 8), [(5, 6), (9, 20), (26, 28), (35, 40)], 0),
            ((30, 35), [(5, 6), (9, 20), (26, 28), (35, 40)], 0),
            # Existence
            ((1, 7), [(5, 6), (9, 20), (26, 28), (35, 40)], 1),
            ((1, 15), [(5, 6), (9, 20), (26, 28), (35, 40)], 1),
            ((6, 10), [(5, 6), (9, 20), (26, 28), (35, 40)], 1),
            ((26, 28), [(5, 6), (9, 20), (26, 28), (35, 40)], 1),
        ],
    )
    def test_existence_reward(self, interval, other_intervals, expected):
        assert _existence_reward(interval, other_intervals) == expected

    @pytest.mark.parametrize(
        "interval,other_intervals,delta,gamma,expected",
        [
            # No overlap
            ((1, 3), [], "flat", "one", 0),
            ((1, 3), [(5, 9)], "flat", "one", 0),
            ((1, 3), [(5, 9), (15, 20)], "flat", "one", 0),
            # Overlap with 1 interval
            ((5, 9), [(5, 9), (15, 20)], "flat", "one", 1),
            (
                (5, 7),
                [(5, 9), (15, 20)],
                "flat",
                "one",
                2 / 2,
            ),  # Same detected and overlap
            (
                (15, 19),
                [(5, 9), (15, 20)],
                "flat",
                "one",
                5 / 5,
            ),  # Same detected and overlap
            ((15, 25), [(5, 9), (15, 20)], "flat", "one", 5 / 10),
            # Overlap with multiple intervals
            ((5, 20), [(5, 9), (15, 20)], "flat", "one", (4 / 15 + 5 / 15) * 1),
            (
                (5, 20),
                [(5, 9), (15, 20)],
                "flat",
                "reciprocal",
                (4 / 15 + 5 / 15) * 1 / 2,
            ),
            (
                (6, 19),
                [(5, 9), (15, 20)],
                "flat",
                "reciprocal",
                (3 / 13 + 4 / 13) * 1 / 2,
            ),
        ],
    )
    def test_overlap_reward(self, interval, other_intervals, delta, gamma, expected):
        assert (
            pytest.approx(_overlap_reward(interval, other_intervals, delta, gamma))
            == expected
        )

    @pytest.mark.parametrize(
        "interval,other_intervals,gamma,expected",
        [
            # No overlap
            ((1, 3), [(5, 6), (9, 20), (26, 28), (35, 40)], "reciprocal", 1),
            ((7, 8), [(5, 6), (9, 20), (26, 28), (35, 40)], "reciprocal", 1),
            ((30, 35), [(5, 6), (9, 20), (26, 28), (35, 40)], "reciprocal", 1),
            # Single overlap
            ((1, 7), [(5, 6), (9, 20), (26, 28), (35, 40)], "reciprocal", 1),
            ((26, 28), [(5, 6), (9, 20), (26, 28), (35, 40)], "reciprocal", 1),
            # Multiple overlap
            ((1, 15), [(5, 6), (9, 20), (26, 28), (35, 40)], "reciprocal", 1 / 2),
            ((5, 10), [(5, 6), (9, 20), (26, 28), (35, 40)], "reciprocal", 1 / 2),
            ((5, 27), [(5, 6), (9, 20), (26, 28), (35, 40)], "reciprocal", 1 / 3),
            ((1, 50), [(5, 6), (9, 20), (26, 28), (35, 40)], "reciprocal", 1 / 4),
        ],
    )
    def test_cardinality_factor(self, interval, other_intervals, gamma, expected):
        assert _cardinality_factor(interval, other_intervals, gamma) == pytest.approx(
            expected
        )

    @pytest.mark.parametrize(
        "interval,ground_truth_intervals,delta,gamma,expected",
        [
            # No overlap
            ((1, 3), [], "flat", "one", 0),
            ((1, 3), [(5, 9)], "flat", "one", 0),
            ((1, 3), [(5, 9), (15, 20)], "flat", "one", 0),
            # Overlap with 1 interval
            ((5, 9), [(5, 9), (15, 20)], "flat", "one", 1),
            (
                (5, 7),
                [(5, 9), (15, 20)],
                "flat",
                "one",
                2 / 2,
            ),  # Same detected and overlap
            (
                (15, 19),
                [(5, 9), (15, 20)],
                "flat",
                "one",
                5 / 5,
            ),  # Same detected and overlap
            ((15, 25), [(5, 9), (15, 20)], "flat", "one", 5 / 10),
            # Overlap with multiple intervals
            ((5, 20), [(5, 9), (15, 20)], "flat", "one", (4 / 15 + 5 / 15) * 1),
            (
                (5, 20),
                [(5, 9), (15, 20)],
                "flat",
                "reciprocal",
                (4 / 15 + 5 / 15) * 1 / 2,
            ),
            (
                (6, 19),
                [(5, 9), (15, 20)],
                "flat",
                "reciprocal",
                (3 / 13 + 4 / 13) * 1 / 2,
            ),
        ],
    )
    def test_precision_interval(
        self, interval, ground_truth_intervals, delta, gamma, expected
    ):
        assert (
            pytest.approx(
                _precision_interval(interval, ground_truth_intervals, delta, gamma)
            )
            == expected
        )

    @pytest.mark.parametrize(
        "interval,predicted_intervals,alpha,delta,gamma,expected",
        [
            # No overlap
            ((1, 3), [], 0.5, "flat", "one", 0),
            ((1, 3), [(5, 9)], 0.5, "flat", "one", 0),
            ((1, 3), [(5, 9), (15, 20)], 0.5, "flat", "one", 0),
            # # Overlap with 1 interval
            ((5, 9), [(5, 9), (15, 20)], 0.0, "flat", "one", 1),
            ((5, 7), [(5, 9), (15, 20)], 0.0, "flat", "one", 2 / 2),
            ((15, 25), [(5, 9), (15, 20)], 0.0, "flat", "one", 5 / 10),
            # # Overlap with multiple intervals
            ((5, 20), [(5, 9), (15, 20)], 0.0, "flat", "one", (4 / 15 + 5 / 15) * 1),
            (
                (5, 20),
                [(5, 9), (15, 20)],
                0.0,
                "flat",
                "reciprocal",
                (4 / 15 + 5 / 15) * 1 / 2,
            ),
            (
                (6, 19),
                [(5, 9), (15, 20)],
                0.0,
                "flat",
                "reciprocal",
                (3 / 13 + 4 / 13) * 1 / 2,
            ),
            # Include existence reward
            ((16, 25), [(5, 9), (15, 20)], 0.25, "flat", "one", 4 / 9 * 0.75 + 0.25),
            (
                (5, 20),
                [(5, 9), (15, 20)],
                0.25,
                "flat",
                "reciprocal",
                ((4 / 15 + 5 / 15) * 1 / 2) * 0.75 + 0.25,
            ),
            (
                (5, 20),
                [(5, 9), (15, 20)],
                0.75,
                "flat",
                "reciprocal",
                ((4 / 15 + 5 / 15) * 1 / 2) * 0.25 + 0.75,
            ),
            ((6, 19), [(5, 9), (15, 20)], 1.0, "flat", "reciprocal", 1.0),
        ],
    )
    def test_recall_interval(
        self, interval, predicted_intervals, alpha, delta, gamma, expected
    ):
        assert (
            pytest.approx(
                _recall_interval(interval, predicted_intervals, alpha, delta, gamma)
            )
            == expected
        )
