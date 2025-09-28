import pytest

from dtaianomaly.type_validation import NoneAttribute


class TestNoneAttribute:

    @pytest.mark.parametrize(
        "value,expected",
        [
            (5, False),
            (1.0, False),
            (True, False),
            ("literal", False),
            ([0, 1, 2, 3], False),
            (None, True),
            ({"a": 1, "b": 2}, False),
        ],
    )
    def test_is_valid_type(self, value, expected):
        assert NoneAttribute()._is_valid_type(value) == expected

    def test_get_valid_type_description(self):
        assert NoneAttribute()._get_valid_type_description() == "None"

    @pytest.mark.parametrize(
        "value,is_valid",
        [
            (5, False),
            (1.0, False),
            (0, False),
            (True, False),
            ("literal", False),
            ([0, 1, 2, 3], False),
            (None, True),
            ({"a": 1, "b": 2}, False),
        ],
    )
    def test_is_valid_value(self, value, is_valid):
        assert NoneAttribute()._is_valid_value(value) == is_valid

    def test_get_valid_value_description(self):
        assert NoneAttribute()._get_valid_value_description() == "None"
