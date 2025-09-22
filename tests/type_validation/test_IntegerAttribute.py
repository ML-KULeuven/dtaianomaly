
import pytest

from dtaianomaly.type_validation import IntegerAttribute


class TestIntegerAttribute:

    def test_initialize_default(self):
        validator = IntegerAttribute()
        assert validator.minimum is None
        assert validator.maximum is None

    @pytest.mark.parametrize('minimum', [None, -1, 0, 1, 42])
    def test_minimum_valid(self, minimum):
        assert IntegerAttribute(minimum=minimum).minimum == minimum

    @pytest.mark.parametrize('minimum', [1.0, True, "auto", [0, 1, 2, 2], {'a': 1, 'b': 2}])
    def test_minimum_invalid(self, minimum):
        with pytest.raises(TypeError):
            IntegerAttribute(minimum=minimum)

    def test_minimum_immutable(self):
        validator = IntegerAttribute(minimum=1)
        with pytest.raises(AttributeError):
            validator.minimum = 2

    @pytest.mark.parametrize('maximum', [None, -1, 0, 1, 42])
    def test_maximum_valid(self, maximum):
        assert IntegerAttribute(maximum=maximum).maximum == maximum

    @pytest.mark.parametrize('maximum', [1.0, True, "auto", [0, 1, 2, 2], {'a': 1, 'b': 2}])
    def test_maximum_invalid(self, maximum):
        with pytest.raises(TypeError):
            IntegerAttribute(maximum=maximum)

    def test_maximum_immutable(self):
        validator = IntegerAttribute(maximum=1)
        with pytest.raises(AttributeError):
            validator.maximum = 2

    @pytest.mark.parametrize("minimum,maximum", [(0, 5), (4, 5), (5, 5)])
    def test_minimum_and_maximum_combined_valid(self, minimum, maximum):
        validator = IntegerAttribute(minimum=minimum, maximum=maximum)
        assert validator.minimum == minimum
        assert validator.maximum == maximum

    @pytest.mark.parametrize("minimum,maximum", [(5, 4)])
    def test_minimum_and_maximum_combined_invalid(self, minimum, maximum):
        with pytest.raises(ValueError):
            IntegerAttribute(minimum=minimum, maximum=maximum)

    @pytest.mark.parametrize("value,expected", [
        (5, True),
        (1.0, False),
        (True, False),
        ("literal", False),
        ([0, 1, 2, 3], False),
        (None, False),
        ({'a': 1, 'b': 2}, False),
    ])
    def test_is_valid_type(self, value, expected):
        assert IntegerAttribute()._is_valid_type(value) == expected

    def test_get_valid_type_description(self):
        assert IntegerAttribute()._get_valid_type_description() == "int"

    @pytest.mark.parametrize("minimum,maximum,value,is_valid", [
        (None, None, 0, True),
        (0, None, 5, True),
        (0, None, 0, True),
        (1, None, 0, False),
        (None, 10, 0, True),
        (None, 0, 0, True),
        (None, -1, 0, False),
        (1, 10, 5, True),
        (1, 10, 1, True),
        (1, 10, 10, True),
        (1, 10, 0, False),
        (1, 10, 11, False),
    ])
    def test_is_valid_value(self, minimum, maximum, value, is_valid):
        assert IntegerAttribute(minimum=minimum, maximum=maximum)._is_valid_value(value) == is_valid

    @pytest.mark.parametrize('minimum', [1])
    def test_get_valid_value_description_only_minimum(self, minimum):
        assert IntegerAttribute(minimum=minimum)._get_valid_value_description() == f"greater than or equal to {minimum}"

    @pytest.mark.parametrize('maximum', [10])
    def test_get_valid_value_description_only_maximum(self, maximum):
        assert IntegerAttribute(maximum=maximum)._get_valid_value_description() == f"less than or equal to {maximum}"

    @pytest.mark.parametrize('minimum', [1])
    @pytest.mark.parametrize('maximum', [10])
    def test_get_valid_value_description_minimum_and_maximum(self, minimum, maximum):
        assert IntegerAttribute(minimum=minimum, maximum=maximum)._get_valid_value_description() == f"in range [{minimum}, {maximum}]"

    def test_get_valid_value_description_no_minimum_and_maximum(self):
        assert IntegerAttribute()._get_valid_value_description() == "int"
