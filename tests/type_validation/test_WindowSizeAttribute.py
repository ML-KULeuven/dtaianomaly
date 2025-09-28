import pytest

from dtaianomaly.type_validation import WindowSizeAttribute
from dtaianomaly.windowing import AUTO_WINDOW_SIZE_COMPUTATION


class TestNoneAttribute:

    @pytest.mark.parametrize(
        "value,expected",
        [
            (5, True),
            (1.0, False),
            (True, False),
            ("literal", True),
            ([0, 1, 2, 3], False),
            (None, False),
            ({"a": 1, "b": 2}, False),
        ],
    )
    def test_is_valid_type(self, value, expected):
        assert WindowSizeAttribute()._is_valid_type(value) == expected

    def test_get_valid_type_description(self):
        assert WindowSizeAttribute()._get_valid_type_description() == "int or string"

    @pytest.mark.parametrize(
        "value,is_valid",
        [
            (1, True),
            (2, True),
            (4, True),
            (10, True),
            (42, True),
            (0, False),
            (-1, False),
            ("auto", False),
        ]
        + [(window_size, True) for window_size in AUTO_WINDOW_SIZE_COMPUTATION],
    )
    def test_is_valid_value(self, value, is_valid):
        assert WindowSizeAttribute()._is_valid_value(value) == is_valid

    def test_get_valid_value_description(self):
        sorted_window_sizes = ", ".join(
            f"'{v}'" for v in sorted(AUTO_WINDOW_SIZE_COMPUTATION)
        )
        assert (
            WindowSizeAttribute()._get_valid_value_description()
            == f"greater than or equal to 1 or in {{{sorted_window_sizes}}}"
        )
