
import pytest

from dtaianomaly.type_validation import FloatAttribute


class TestFloatAttribute:

    def test_initialize_default(self):
        validator = FloatAttribute()
        assert validator.minimum is None
        assert validator.maximum is None
        assert validator.inclusive_minimum
        assert validator.inclusive_maximum

    @pytest.mark.parametrize('minimum', [None, -1.0, 0.5, 1.3, 42.123456789])
    def test_minimum_valid(self, minimum):
        assert FloatAttribute(minimum=minimum).minimum == minimum

    @pytest.mark.parametrize('minimum', [1, True, "auto", [0, 1, 2, 2], {'a': 1, 'b': 2}])
    def test_minimum_invalid(self, minimum):
        with pytest.raises(TypeError):
            FloatAttribute(minimum=minimum)

    def test_minimum_immutable(self):
        validator = FloatAttribute(minimum=1.5)
        with pytest.raises(AttributeError):
            validator.minimum = 2

    @pytest.mark.parametrize('maximum', [None, -1.0, 0.5, 1.3, 42.123456789])
    def test_maximum_valid(self, maximum):
        assert FloatAttribute(maximum=maximum).maximum == maximum

    @pytest.mark.parametrize('maximum', [1, True, "auto", [0, 1, 2, 2], {'a': 1, 'b': 2}])
    def test_maximum_invalid(self, maximum):
        with pytest.raises(TypeError):
            FloatAttribute(maximum=maximum)

    def test_maximum_immutable(self):
        validator = FloatAttribute(maximum=1.5)
        with pytest.raises(AttributeError):
            validator.maximum = 2

    @pytest.mark.parametrize("minimum,maximum", [(0.1, 3.14), (3.14, 3.14)])
    def test_minimum_and_maximum_combined_valid(self, minimum, maximum):
        validator = FloatAttribute(minimum=minimum, maximum=maximum)
        assert validator.minimum == minimum
        assert validator.maximum == maximum

    @pytest.mark.parametrize("minimum,maximum", [(3.15, 3.14)])
    def test_minimum_and_maximum_combined_invalid(self, minimum, maximum):
        with pytest.raises(ValueError):
            FloatAttribute(minimum=minimum, maximum=maximum)

    @pytest.mark.parametrize('inclusive_minimum', [True, False])
    def test_inclusive_minimum_valid(self, inclusive_minimum):
        assert FloatAttribute(inclusive_minimum=inclusive_minimum).inclusive_minimum == inclusive_minimum

    @pytest.mark.parametrize('inclusive_minimum', [1, 1.5, "auto", [0, 1, 2, 2], {'a': 1, 'b': 2}])
    def test_inclusive_minimum_invalid(self, inclusive_minimum):
        with pytest.raises(TypeError):
            FloatAttribute(inclusive_minimum=inclusive_minimum)

    def test_inclusive_minimum_immutable(self):
        validator = FloatAttribute(inclusive_minimum=True)
        with pytest.raises(AttributeError):
            validator.inclusive_minimum = False

    @pytest.mark.parametrize('inclusive_maximum', [True, False])
    def test_inclusive_maximum_valid(self, inclusive_maximum):
        assert FloatAttribute(inclusive_maximum=inclusive_maximum).inclusive_maximum == inclusive_maximum

    @pytest.mark.parametrize('inclusive_maximum', [1, 1.5, "auto", [0, 1, 2, 2], {'a': 1, 'b': 2}])
    def test_inclusive_maximum_invalid(self, inclusive_maximum):
        with pytest.raises(TypeError):
            FloatAttribute(inclusive_maximum=inclusive_maximum)

    def test_inclusive_maximum_immutable(self):
        validator = FloatAttribute(inclusive_maximum=True)
        with pytest.raises(AttributeError):
            validator.inclusive_maximum = False

    @pytest.mark.parametrize("value,expected", [
        (5, False),
        (1.0, True),
        (True, False),
        ("literal", False),
        ([0, 1, 2, 3], False),
        (None, False),
        ({'a': 1, 'b': 2}, False),
    ])
    def test_is_valid_type(self, value, expected):
        assert FloatAttribute()._is_valid_type(value) == expected

    def test_get_valid_type_description(self):
        assert FloatAttribute()._get_valid_type_description() == "float"

    @pytest.mark.parametrize("minimum,maximum,inclusive_minimum,inclusive_maximum,value,is_valid", [
        (None, None, True, True, 0.0, True),
        (0.3, None, True, True, 5.6, True),
        (0.3, None, True, True, 0.3, True),
        (0.3, None, False, True, 0.3, False),
        (0.3, None, True, True, -1.0, False),
        (None, 10.1, True, True, 0.0, True),
        (None, 1.0, True, True, 0.99999999999, True),
        (None, 1.0, True, True, 1.0, True),
        (None, 1.0, True, False, 1.0, False),
        (None, -1.0, True, True, 0.0, False),
        (1.3, 10.9, True, True, 5.0, True),
        (1.3, 10.9, True, True, 1.3, True),
        (1.3, 10.9, False, True, 1.3, False),
        (1.3, 10.9, True, True, 10.9, True),
        (1.3, 10.9, True, False, 10.9, False),
        (1.3, 10.9, True, True, 0.0, False),
        (1.3, 10.9, True, True, 11.0, False),
    ])
    def test_is_valid_value(self, minimum, maximum, inclusive_minimum, inclusive_maximum, value, is_valid):
        assert FloatAttribute(minimum=minimum, maximum=maximum, inclusive_minimum=inclusive_minimum, inclusive_maximum=inclusive_maximum)._is_valid_value(value) == is_valid

    @pytest.mark.parametrize('minimum', [1.1])
    def test_get_valid_value_description_only_minimum_inclusive(self, minimum):
        assert FloatAttribute(minimum=minimum, inclusive_minimum=True)._get_valid_value_description() == f"greater than or equal to {minimum}"

    @pytest.mark.parametrize('minimum', [1.1])
    def test_get_valid_value_description_only_minimum_exclusive(self, minimum):
        assert FloatAttribute(minimum=minimum, inclusive_minimum=False)._get_valid_value_description() == f"greater than {minimum}"

    @pytest.mark.parametrize('maximum', [10.0])
    def test_get_valid_value_description_only_maximum_inclusive(self, maximum):
        assert FloatAttribute(maximum=maximum, inclusive_maximum=True)._get_valid_value_description() == f"less than or equal to {maximum}"

    @pytest.mark.parametrize('maximum', [10.0])
    def test_get_valid_value_description_only_maximum_exclusive(self, maximum):
        assert FloatAttribute(maximum=maximum, inclusive_maximum=False)._get_valid_value_description() == f"less than {maximum}"

    @pytest.mark.parametrize('minimum', [1.1])
    @pytest.mark.parametrize('maximum', [10.0])
    def test_get_valid_value_description_minimum_and_maximum_inclusive_inclusive(self, minimum, maximum):
        assert FloatAttribute(minimum=minimum, maximum=maximum, inclusive_minimum=True, inclusive_maximum=True)._get_valid_value_description() == f"in range [{minimum}, {maximum}]"

    @pytest.mark.parametrize('minimum', [1.1])
    @pytest.mark.parametrize('maximum', [10.0])
    def test_get_valid_value_description_minimum_and_maximum_inclusive_exclusive(self, minimum, maximum):
        assert FloatAttribute(minimum=minimum, maximum=maximum, inclusive_minimum=True, inclusive_maximum=False)._get_valid_value_description() == f"in range [{minimum}, {maximum}["

    @pytest.mark.parametrize('minimum', [1.1])
    @pytest.mark.parametrize('maximum', [10.0])
    def test_get_valid_value_description_minimum_and_maximum_exclusive_inclusive(self, minimum, maximum):
        assert FloatAttribute(minimum=minimum, maximum=maximum, inclusive_minimum=False, inclusive_maximum=True)._get_valid_value_description() == f"in range ]{minimum}, {maximum}]"

    @pytest.mark.parametrize('minimum', [1.1])
    @pytest.mark.parametrize('maximum', [10.0])
    def test_get_valid_value_description_minimum_and_maximum_exclusive_exclusive(self, minimum, maximum):
        assert FloatAttribute(minimum=minimum, maximum=maximum, inclusive_minimum=False, inclusive_maximum=False)._get_valid_value_description() == f"in range ]{minimum}, {maximum}["

    def test_get_valid_value_description_no_minimum_and_maximum(self):
        assert FloatAttribute()._get_valid_value_description() == "float"
