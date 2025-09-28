import pytest
from conftest import ATTRIBUTE_VALIDATION_CONFIGS

from dtaianomaly.type_validation import AttributeValidationMixin, IntegerAttribute


class TestAttributeValidationMixin:

    @pytest.mark.parametrize("config", ATTRIBUTE_VALIDATION_CONFIGS)
    def test(self, config):

        class MyObject(AttributeValidationMixin):
            attribute: any
            attribute_validation = {"attribute": config["validator"]}

            def __init__(self, attribute):
                self.attribute = attribute

        valid = config["valid"]
        invalid_type = config["invalid_type"]
        invalid_value = config["invalid_value"]
        expected_value_error = (
            ValueError
            if config["validator"]._is_valid_type(invalid_value)
            else TypeError
        )

        # Cannot initialize with invalid type
        with pytest.raises(TypeError):
            MyObject(invalid_type)

        # Cannot initialize with invalid value
        with pytest.raises(expected_value_error):
            MyObject(invalid_value)

        # Cannot initialize with valid value
        my_object = MyObject(valid)
        assert my_object.attribute == valid

        # Cannot update to invalid type
        with pytest.raises(TypeError):
            my_object.attribute = invalid_type

        # Cannot update to invalid value
        with pytest.raises(expected_value_error):
            my_object.attribute = invalid_value

        # Can update to valid value
        my_object.attribute = valid

    @pytest.mark.parametrize("config", ATTRIBUTE_VALIDATION_CONFIGS)
    def test_no_validation(self, config):

        class MyObject(AttributeValidationMixin):
            attribute: any

            def __init__(self, attribute):
                self.attribute = attribute

        valid = config["valid"]
        invalid_type = config["invalid_type"]
        invalid_value = config["invalid_value"]

        assert MyObject(invalid_type).attribute == invalid_type
        assert MyObject(invalid_value).attribute == invalid_value
        assert MyObject(valid).attribute == valid

        my_object = MyObject(valid)

        for value in [invalid_type, invalid_value, valid]:
            # You can initialize the object
            assert MyObject(value).attribute == value

            # The variable is mutable
            my_object.attribute = value
            assert my_object.attribute == value

    @pytest.mark.parametrize("config_parent", ATTRIBUTE_VALIDATION_CONFIGS)
    @pytest.mark.parametrize("config_child", ATTRIBUTE_VALIDATION_CONFIGS)
    def test_inheritance(self, config_parent, config_child):

        class Parent(AttributeValidationMixin):
            attribute_parent: any
            attribute_validation = {"attribute_parent": config_parent["validator"]}

            def __init__(self, attribute_parent):
                self.attribute_parent = attribute_parent

        class Child(Parent):
            attribute_child: any
            attribute_validation = {"attribute_child": config_child["validator"]}

            def __init__(self, attribute_parent, attribute_child):
                super().__init__(attribute_parent=attribute_parent)
                self.attribute_child = attribute_child

        # Check if a value error or type error will be raised for invalid values
        expected_value_error_parent = (
            ValueError
            if config_parent["validator"]._is_valid_type(config_parent["invalid_value"])
            else TypeError
        )
        expected_value_error_child = (
            ValueError
            if config_child["validator"]._is_valid_type(config_child["invalid_value"])
            else TypeError
        )

        # Cannot initialize if any has invalid type
        with pytest.raises(TypeError):
            Child(config_parent["invalid_type"], config_child["invalid_type"])
        with pytest.raises(TypeError):
            Child(config_parent["valid"], config_child["invalid_type"])
        with pytest.raises(TypeError):
            Child(config_parent["invalid_type"], config_child["valid"])

        # Cannot initialize with invalid value
        with pytest.raises(expected_value_error_parent):
            Child(config_parent["invalid_value"], config_child["invalid_value"])
        with pytest.raises(expected_value_error_child):
            Child(config_parent["valid"], config_child["invalid_value"])
        with pytest.raises(expected_value_error_parent):
            Child(config_parent["invalid_value"], config_child["valid"])

        # Cannot initialize with valid value
        child = Child(config_parent["valid"], config_child["valid"])
        assert child.attribute_parent == config_parent["valid"]
        assert child.attribute_child == config_child["valid"]

        # Cannot update to invalid type
        with pytest.raises(TypeError):
            child.attribute_parent = config_parent["invalid_type"]
        with pytest.raises(TypeError):
            child.attribute_child = config_child["invalid_type"]

        # Cannot update to invalid value
        with pytest.raises(expected_value_error_parent):
            child.attribute_parent = config_parent["invalid_value"]
        with pytest.raises(expected_value_error_child):
            child.attribute_child = config_child["invalid_value"]

        # Can update to valid value
        child.attribute_parent = config_parent["valid"]
        child.attribute_child = config_child["valid"]

    def test_immutable_validation(self):

        class MyObject(AttributeValidationMixin):
            attribute: int
            attribute_validation = {"attribute": IntegerAttribute()}

            def __init__(self, attribute):
                self.attribute = attribute

        my_object = MyObject(5)
        with pytest.raises(AttributeError):
            my_object.attribute_validation = {}

    def test_immutable_validation_inheritance(self):

        class Parent(AttributeValidationMixin):
            attribute_parent: any
            attribute_validation = {"attribute_parent": IntegerAttribute(minimum=5)}

            def __init__(self, attribute_parent):
                self.attribute_parent = attribute_parent

        class Child(Parent):
            attribute_child: any
            attribute_validation = {"attribute_child": IntegerAttribute(maximum=5)}

            def __init__(self, attribute_parent, attribute_child):
                super().__init__(attribute_parent=attribute_parent)
                self.attribute_child = attribute_child

        child = Child(5, 5)
        with pytest.raises(AttributeError):
            child.attribute_validation = {}
