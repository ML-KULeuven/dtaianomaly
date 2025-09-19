import pytest

from dtaianomaly.type_validation import BoolAttribute


class TestBoolAttribute:

    def test_initialize_default(self):
        BoolAttribute()   # Does not raise an error

    @pytest.mark.parametrize("value,expected", [
        (5, False),
        (1.0, False),
        (True, True),
        (False, True),
        ("literal", False),
        ([0, 1, 2, 3], False),
        (None, False),
        ({'a': 1, 'b': 2}, False),
    ])
    def test_is_valid_type(self, value, expected):
        assert BoolAttribute()._is_valid_type(value) == expected

    def test_get_valid_type_description(self):
        assert BoolAttribute()._get_valid_type_description() == "bool"

    @pytest.mark.parametrize('value,is_valid', [
        (1, False),
        (0, False),
        (5, False),
        (0.3, False),
        (0.0, False),
        (1.0, False),
        (True, True),
        (False, True),
        ("auto", False),
        ([0, 1, 2, 2], False),
        ([], False),
        ({'a': 1, 'b': 2}, False),
        ({}, False)
    ])
    def test_is_valid_value(self, value, is_valid):
        assert BoolAttribute()._is_valid_value(value) == is_valid

    def test_get_valid_value_description(self):
        assert BoolAttribute()._get_valid_value_description() == "True or False"
