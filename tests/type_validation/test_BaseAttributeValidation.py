
import pytest
from conftest import ATTRIBUTE_VALIDATION_CONFIGS

from dtaianomaly.type_validation import IntegerAttribute, LiteralAttribute, NoneAttribute, UnionAttribute


class TestBaseAttributeValidation:

    @pytest.mark.parametrize("config", ATTRIBUTE_VALIDATION_CONFIGS)
    def test_raise_error_if_invalid_no_error(self, config):
        config['validator'].raise_error_if_invalid(config['valid'], "my_attribute", "MyClass")

    @pytest.mark.parametrize("config", ATTRIBUTE_VALIDATION_CONFIGS)
    def test_raise_error_if_invalid_type(self, config):
        with pytest.raises(TypeError):
            config['validator'].raise_error_if_invalid(config['invalid_type'], "my_attribute", "MyClass")

    @pytest.mark.parametrize("config", ATTRIBUTE_VALIDATION_CONFIGS)
    def test_raise_error_if_invalid_value(self, config):
        expected_error = ValueError if config['validator']._is_valid_type(config['invalid_value']) else TypeError
        with pytest.raises(expected_error):
            config['validator'].raise_error_if_invalid(config['invalid_value'], "my_attribute", "MyClass")

    def test_or(self):
        validation = IntegerAttribute(minimum=5) | LiteralAttribute('auto') | NoneAttribute()

        assert isinstance(validation, UnionAttribute)
        assert len(validation.attribute_validators) == 3

        assert isinstance(validation.attribute_validators[0], IntegerAttribute)
        assert validation.attribute_validators[0].minimum == 5
        assert validation.attribute_validators[0].maximum is None

        assert isinstance(validation.attribute_validators[1], LiteralAttribute)
        assert len(validation.attribute_validators[1].values) == 1
        assert list(validation.attribute_validators[1].values)[0] == "auto"

        assert isinstance(validation.attribute_validators[2], NoneAttribute)

    def test_or_attribute_validation_with_other(self):
        with pytest.raises(TypeError):
            IntegerAttribute(minimum=5) | 5

    def test_or_other_with_attribute_validation(self):
        with pytest.raises(TypeError):
            5 | IntegerAttribute(minimum=5)

    def test_or_multiple_attribute_validation_with_other(self):
        with pytest.raises(TypeError):
            IntegerAttribute(minimum=5) | LiteralAttribute('auto') | NoneAttribute() | 5

