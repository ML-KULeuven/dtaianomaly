import pytest
from conftest import ATTRIBUTE_VALIDATION_CONFIGS

from dtaianomaly.type_validation import (
    AttributeValidationMixin,
    IntegerAttribute,
    ListAttribute,
    LiteralAttribute,
    NoneAttribute,
    UnionAttribute,
)

# None and Union validators have a dedicated value description, tested separately
_CONFIGS_WITH_SIMPLE_VALUE_DESCRIPTION = [
    config
    for config in ATTRIBUTE_VALIDATION_CONFIGS
    if not isinstance(config["validator"], (NoneAttribute, UnionAttribute))
]


class TestListAttribute:

    def test_initialize_default(self):
        with pytest.raises(TypeError):
            ListAttribute()

    @pytest.mark.parametrize("config", ATTRIBUTE_VALIDATION_CONFIGS)
    def test_validator_valid(self, config):
        assert isinstance(
            ListAttribute(config["validator"]).validator, config["validator"].__class__
        )

    @pytest.mark.parametrize(
        "validator", [5, 1.0, True, "auto", [0, 1, 2, 2], {"a": 1, "b": 2}]
    )
    def test_validator_invalid(self, validator):
        with pytest.raises(TypeError):
            ListAttribute(validator)

    def test_validator_immutable(self):
        validator = ListAttribute(IntegerAttribute())
        with pytest.raises(AttributeError):
            validator.validator = NoneAttribute()

    @pytest.mark.parametrize("minimum_length", [0, 1, 2, 3, 4, 5])
    def test_minimum_length_valid(self, minimum_length):
        assert (
            ListAttribute(
                IntegerAttribute(), minimum_length=minimum_length
            ).minimum_length
            == minimum_length
        )

    @pytest.mark.parametrize(
        "minimum_length", [1.0, True, "auto", [0, 1, 2, 2], {"a": 1, "b": 2}]
    )
    def test_minimum_length_invalid_type(self, minimum_length):
        with pytest.raises(TypeError):
            ListAttribute(IntegerAttribute(), minimum_length=minimum_length)

    @pytest.mark.parametrize("minimum_length", [-1])
    def test_minimum_length_invalid_value(self, minimum_length):
        with pytest.raises(ValueError):
            ListAttribute(IntegerAttribute(), minimum_length=minimum_length)

    def test_minimum_length_immutable(self):
        validator = ListAttribute(IntegerAttribute())
        with pytest.raises(AttributeError):
            validator.minimum_length = 10

    @pytest.mark.parametrize("maximum_length", [1, 2, 3, 4, 5])
    def test_maximum_length_valid(self, maximum_length):
        assert (
            ListAttribute(
                IntegerAttribute(), maximum_length=maximum_length
            ).maximum_length
            == maximum_length
        )

    @pytest.mark.parametrize(
        "maximum_length", [1.0, True, "auto", [0, 1, 2, 2], {"a": 1, "b": 2}]
    )
    def test_maximum_length_invalid_type(self, maximum_length):
        with pytest.raises(TypeError):
            ListAttribute(IntegerAttribute(), maximum_length=maximum_length)

    @pytest.mark.parametrize("maximum_length", [0, -1])
    def test_maximum_length_invalid_value(self, maximum_length):
        with pytest.raises(ValueError):
            ListAttribute(IntegerAttribute(), maximum_length=maximum_length)

    def test_maximum_length_immutable(self):
        validator = ListAttribute(IntegerAttribute())
        with pytest.raises(AttributeError):
            validator.maximum_length = 10

    @pytest.mark.parametrize("minimum_length,maximum_length", [(0, 5), (4, 5), (5, 5)])
    def test_minimum_and_maximum_combined_valid(self, minimum_length, maximum_length):
        validator = ListAttribute(
            IntegerAttribute(),
            minimum_length=minimum_length,
            maximum_length=maximum_length,
        )
        assert validator.minimum_length == minimum_length
        assert validator.maximum_length == maximum_length

    @pytest.mark.parametrize("minimum_length,maximum_length", [(5, 4)])
    def test_minimum_and_maximum_combined_invalid(self, minimum_length, maximum_length):
        with pytest.raises(ValueError):
            ListAttribute(
                IntegerAttribute(),
                minimum_length=minimum_length,
                maximum_length=maximum_length,
            )

    @pytest.mark.parametrize("config", ATTRIBUTE_VALIDATION_CONFIGS)
    def test_is_valid_type(self, config):
        # Valid
        assert ListAttribute(config["validator"])._is_valid_type([config["valid"]])
        assert ListAttribute(config["validator"])._is_valid_type(5 * [config["valid"]])

        # Invalid
        assert not ListAttribute(config["validator"])._is_valid_type(
            config["valid"]
        )  # Not a list of the requested type
        assert not ListAttribute(config["validator"])._is_valid_type(
            [config["invalid_type"]]
        )
        assert not ListAttribute(config["validator"])._is_valid_type(
            5 * [config["invalid_type"]]
        )
        one_invalid = 5 * [config["valid"]]
        one_invalid[3] = config["invalid_type"]
        assert not ListAttribute(config["validator"])._is_valid_type(one_invalid)

    @pytest.mark.parametrize("config", ATTRIBUTE_VALIDATION_CONFIGS)
    def test_get_valid_type_description(self, config):
        assert (
            ListAttribute(config["validator"])._get_valid_type_description()
            == f"list of {config['validator']._get_valid_type_description()}"
        )

    def test_get_valid_type_description_defined(self):
        assert (
            ListAttribute(IntegerAttribute())._get_valid_type_description()
            == "list of int"
        )

    @pytest.mark.parametrize("config", ATTRIBUTE_VALIDATION_CONFIGS)
    def test_is_valid_value(self, config):
        # Valid
        assert ListAttribute(config["validator"])._is_valid_value([config["valid"]])
        assert ListAttribute(config["validator"])._is_valid_value(5 * [config["valid"]])

        # Invalid
        assert not ListAttribute(config["validator"])._is_valid_value(
            [config["invalid_value"]]
        )
        assert not ListAttribute(config["validator"])._is_valid_value(
            5 * [config["invalid_value"]]
        )
        one_invalid = 5 * [config["valid"]]
        one_invalid[3] = config["invalid_value"]
        assert not ListAttribute(config["validator"])._is_valid_value(one_invalid)

    @pytest.mark.parametrize(
        "minimum_length,maximum_length,length,expected",
        [
            (5, None, 5, True),
            (5, None, 4, False),
            (None, 5, 5, True),
            (None, 5, 6, False),
            (5, 5, 5, True),
            (5, 10, 7, True),
            (5, 10, 4, False),
            (5, 10, 11, False),
        ],
    )
    def test_is_valid_value_length(
        self, minimum_length, maximum_length, length, expected
    ):
        validator = ListAttribute(
            IntegerAttribute(minimum=1),
            minimum_length=minimum_length,
            maximum_length=maximum_length,
        )
        assert validator._is_valid_value(length * [1]) == expected

    @pytest.mark.parametrize("config", _CONFIGS_WITH_SIMPLE_VALUE_DESCRIPTION)
    def test_get_valid_value_description(self, config):
        assert (
            ListAttribute(config["validator"])._get_valid_value_description()
            == f"list of {config['validator']._get_valid_type_description()} {config['validator']._get_valid_value_description()}"
        )

    def test_get_valid_value_description_defined(self):
        assert (
            ListAttribute(IntegerAttribute(minimum=5))._get_valid_value_description()
            == "list of int greater than or equal to 5"
        )

    def test_get_valid_value_description_none(self):
        assert (
            ListAttribute(NoneAttribute())._get_valid_value_description()
            == "list of None"
        )

    def test_get_valid_value_description_union(self):
        assert (
            ListAttribute(
                IntegerAttribute(minimum=5) | LiteralAttribute("auto")
            )._get_valid_value_description()
            == "list of int greater than or equal to 5 or string in {'auto'}"
        )

    def test_get_valid_value_description_union_3(self):
        assert (
            ListAttribute(
                IntegerAttribute(minimum=5) | LiteralAttribute("auto") | NoneAttribute()
            )._get_valid_value_description()
            == "list of int greater than or equal to 5, string in {'auto'} or None"
        )

    def test_get_valid_value_description_minimum_length(self):
        assert (
            ListAttribute(
                IntegerAttribute(), minimum_length=5
            )._get_valid_value_description()
            == "list of int int with minimum 5 elements"
        )

    def test_get_valid_value_description_minimum_length_0(self):
        assert (
            ListAttribute(
                IntegerAttribute(), minimum_length=0
            )._get_valid_value_description()
            == "list of int int"
        )

    def test_get_valid_value_description_maximum_length(self):
        assert (
            ListAttribute(
                IntegerAttribute(), maximum_length=5
            )._get_valid_value_description()
            == "list of int int with maximum 5 elements"
        )

    def test_get_valid_value_description_minimum_and_maximum_length(self):
        assert (
            ListAttribute(
                IntegerAttribute(), minimum_length=3, maximum_length=5
            )._get_valid_value_description()
            == "list of int int with minimum 3 elements and maximum 5 elements"
        )

    def test_get_valid_value_description_minimum_and_maximum_length_equal(self):
        assert (
            ListAttribute(
                IntegerAttribute(), minimum_length=5, maximum_length=5
            )._get_valid_value_description()
            == "list of int int with 5 elements"
        )

    def test_args(self):

        class MyObject(AttributeValidationMixin):
            integers: list[int]
            attribute_validation = {"integers": ListAttribute(IntegerAttribute())}

            def __init__(self, *integers: int):
                self.integers = list(integers)

        my_object = MyObject(1, 2, 3, 4, 5)
        assert len(my_object.integers) == 5
