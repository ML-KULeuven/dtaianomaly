import pytest

from dtaianomaly.utils import PrintConstructionCallMixin

properties = [
    (int, 5, 1),
    # (str, "a string", "another string"),  # Test separate because otherwise it is annoying with the apostrophes
    (float, 0.5, 3.14),
    (bool, True, False),
    (list[int], [1, 2, 3], [4, 5, 6, 7]),
    (dict[str, int], {"a": 1, "b": 2}, {"x": 100, "y": 200, "z": 300}),
]

arg_properties = [
    (int, (1, 2, 3)),
    (int, (1, 2, 3, 4, 5, 6)),
    (float, (0.1, 0.2, 0.4, 0.8, 1.6)),
]

kwarg_properties = [
    (int, {"a": 1, "b": 2}),
    (float, {"x": 1.414, "y": 2.718, "z": 3.14}),
]


class TestNoParametersPrintConstructionCallMixin:

    def test(self):
        class MyObject(PrintConstructionCallMixin):
            def __init__(self):
                pass

        assert str(MyObject()) == "MyObject()"

    def test_no_constructor(self):
        class MyObject(PrintConstructionCallMixin):
            pass

        assert str(MyObject()) == "MyObject()"

    def test_inheritance(self):
        class Parent(PrintConstructionCallMixin):
            pass

        class Child(Parent):
            pass

        assert str(Child()) == f"Child()"

    def test_inheritance_constructor_child(self):
        class Parent(PrintConstructionCallMixin):
            pass

        class Child(Parent):
            def __init__(self):
                pass

        assert str(Child()) == f"Child()"

    def test_inheritance_constructor_parent(self):
        class Parent(PrintConstructionCallMixin):
            def __init__(self):
                pass

        class Child(Parent):
            pass

        assert str(Child()) == f"Child()"


class TestStrPrintConstructionCallMixin:

    def test_no_default(self):
        class MyObject(PrintConstructionCallMixin):
            def __init__(self, a: str):
                self.a = a

        assert str(MyObject("a-string")) == f"MyObject(a='a-string')"

    def test_default(self):
        class MyObject(PrintConstructionCallMixin):
            def __init__(self, a: str = "default"):
                self.a = a

        assert str(MyObject()) == f"MyObject()"
        assert str(MyObject(a="default")) == f"MyObject()"

    def test_default_other(self):
        class MyObject(PrintConstructionCallMixin):
            def __init__(self, a: str = "default"):
                self.a = a

        assert str(MyObject(a="another-one")) == f"MyObject(a='another-one')"


@pytest.mark.parametrize("attribute_type,attribute_default,other_value", properties)
class TestBaseCasesPrintConstructionCallMixin:

    def test_no_default(self, attribute_type, attribute_default, other_value):

        class MyObject(PrintConstructionCallMixin):
            def __init__(self, a: attribute_type):
                self.a = a

        assert str(MyObject(other_value)) == f"MyObject(a={other_value})"

    def test_default(self, attribute_type, attribute_default, other_value):

        class MyObject(PrintConstructionCallMixin):
            def __init__(self, a: attribute_type = attribute_default):
                self.a = a

        assert str(MyObject()) == f"MyObject()"
        assert str(MyObject(a=attribute_default)) == f"MyObject()"

    def test_default_other(self, attribute_type, attribute_default, other_value):

        class MyObject(PrintConstructionCallMixin):
            def __init__(self, a: attribute_type = attribute_default):
                self.a = a

        assert str(MyObject(other_value)) == f"MyObject(a={other_value})"

    def test_different_name(self, attribute_type, attribute_default, other_value):

        class MyObject(PrintConstructionCallMixin):
            def __init__(self, a: attribute_type = attribute_default):
                self.a_ = a

        with pytest.raises(AttributeError):
            str(MyObject(other_value))

    def test_inheritance(self, attribute_type, attribute_default, other_value):

        class Parent(PrintConstructionCallMixin):
            pass

        class Child(Parent):
            def __init__(self, a: attribute_type = attribute_default):
                self.a = a

        assert str(Child()) == f"Child()"
        assert str(Child(a=attribute_default)) == f"Child()"
        assert str(Child(other_value)) == f"Child(a={other_value})"

    def test_inheritance_constructor_in_parent(
        self, attribute_type, attribute_default, other_value
    ):

        class Parent(PrintConstructionCallMixin):
            def __init__(self, a: attribute_type = attribute_default):
                self.a = a

        class Child(Parent):
            pass

        assert str(Child()) == f"Child()"
        assert str(Child(a=attribute_default)) == f"Child()"
        assert str(Child(other_value)) == f"Child(a={other_value})"


@pytest.mark.parametrize("attribute_type1,attribute_default1,other_value1", properties)
@pytest.mark.parametrize("attribute_type2,attribute_default2,other_value2", properties)
class TestPairwisePrintConstructionCallMixin:

    def test(
        self,
        attribute_type1,
        attribute_default1,
        other_value1,
        attribute_type2,
        attribute_default2,
        other_value2,
    ):

        class MyObject(PrintConstructionCallMixin):
            def __init__(
                self,
                a: attribute_type1 = attribute_default1,
                b: attribute_type2 = attribute_default2,
            ):
                self.a = a
                self.b = b

        assert str(MyObject()) == f"MyObject()"
        assert str(MyObject(a=other_value1)) == f"MyObject(a={other_value1})"
        assert str(MyObject(b=other_value2)) == f"MyObject(b={other_value2})"
        assert (
            str(MyObject(a=other_value1, b=other_value2))
            == f"MyObject(a={other_value1},b={other_value2})"
        )


class TestObjectParameters:

    def test_with_mixin_parameter(self):

        class Other(PrintConstructionCallMixin):
            def __init__(self):
                pass

        class MyObject(PrintConstructionCallMixin):
            def __init__(self, other: Other):
                self.other = other

        assert str(MyObject(Other())) == f"MyObject(other=Other())"

    @pytest.mark.parametrize("attribute_type,attribute_default,other_value", properties)
    def test_with_parametrized_mixin_parameter(
        self, attribute_type, attribute_default, other_value
    ):

        class Other(PrintConstructionCallMixin):
            def __init__(self, a: attribute_type = attribute_default):
                self.a = a

        class MyObject(PrintConstructionCallMixin):
            def __init__(self, other: Other):
                self.other = other

        assert str(MyObject(Other())) == f"MyObject(other=Other())"
        assert str(MyObject(Other(a=attribute_default))) == f"MyObject(other=Other())"
        assert (
            str(MyObject(Other(a=other_value)))
            == f"MyObject(other=Other(a={other_value}))"
        )

    def test_without_mixin_parameter(self):

        class Other:
            pass

        class MyObject(PrintConstructionCallMixin):
            def __init__(self, other: Other):
                self.other = other

        my_object = MyObject(Other())
        with pytest.raises(AttributeError):
            str(my_object)


@pytest.mark.parametrize("arg_type,values", arg_properties)
class TestArgParameters:

    def test(self, arg_type, values):

        class MyObject(PrintConstructionCallMixin):
            def __init__(self, *a: arg_type):
                self.a = a

        assert str(MyObject(*values)) == f"MyObject({','.join(map(str, values))})"

    @pytest.mark.parametrize("attribute_type,attribute_default,other_value", properties)
    def test_with_other_property(
        self, attribute_type, attribute_default, other_value, arg_type, values
    ):

        class MyObject(PrintConstructionCallMixin):
            def __init__(self, other: attribute_type = attribute_default, *a: arg_type):
                self.other = other
                self.a = a

        assert (
            str(MyObject(other_value, *values))
            == f"MyObject({other_value},{','.join(map(str, values))})"
        )
        assert str(MyObject(other_value)) == f"MyObject(other={other_value})"

    @pytest.mark.parametrize(
        "attribute_type1,attribute_default1,other_value1", properties
    )
    @pytest.mark.parametrize(
        "attribute_type2,attribute_default2,other_value2", properties
    )
    def test_with_two_other_property(
        self,
        attribute_type1,
        attribute_default1,
        other_value1,
        attribute_type2,
        attribute_default2,
        other_value2,
        arg_type,
        values,
    ):

        class MyObject(PrintConstructionCallMixin):
            def __init__(
                self,
                other1: attribute_type1 = attribute_default1,
                other2: attribute_type2 = attribute_default2,
                *a: arg_type,
            ):
                self.other1 = other1
                self.other2 = other2
                self.a = a

        assert (
            str(MyObject(other_value1, other_value2, *values))
            == f"MyObject({other_value1},{other_value2},{','.join(map(str, values))})"
        )
        assert (
            str(MyObject(other_value1, other_value2))
            == f"MyObject(other1={other_value1},other2={other_value2})"
        )

    @pytest.mark.parametrize("attribute_type,attribute_default,other_value", properties)
    def test_first_args_then_other(
        self, attribute_type, attribute_default, other_value, arg_type, values
    ):
        class MyObject(PrintConstructionCallMixin):
            def __init__(self, *a: arg_type, other: attribute_type = attribute_default):
                self.other = other
                self.a = a

        assert (
            str(MyObject(*values, other=other_value))
            == f"MyObject({','.join(map(str, values))},other={other_value})"
        )
        assert str(MyObject(other=other_value)) == f"MyObject(other={other_value})"
        assert str(MyObject(*values)) == f"MyObject({','.join(map(str, values))})"


@pytest.mark.parametrize("kwarg_type,values", kwarg_properties)
class TestKwargParameters:

    def test(self, kwarg_type, values):
        class MyObject(PrintConstructionCallMixin):
            def __init__(self, **a: kwarg_type):
                self.a = a

        assert (
            str(MyObject(**values))
            == f"MyObject({','.join(map(lambda k: f'{k}={values[k]}', values))})"
        )

    @pytest.mark.parametrize("attribute_type,attribute_default,other_value", properties)
    def test_with_other_property(
        self, attribute_type, attribute_default, other_value, kwarg_type, values
    ):

        class MyObject(PrintConstructionCallMixin):
            def __init__(
                self, other: attribute_type = attribute_default, **a: kwarg_type
            ):
                self.other = other
                self.a = a

        assert (
            str(MyObject(other_value, **values))
            == f"MyObject(other={other_value},{','.join(map(lambda k: f'{k}={values[k]}', values))})"
        )
        assert str(MyObject(other_value)) == f"MyObject(other={other_value})"
        assert (
            str(MyObject(**values))
            == f"MyObject({','.join(map(lambda k: f'{k}={values[k]}', values))})"
        )

    @pytest.mark.parametrize(
        "attribute_type1,attribute_default1,other_value1", properties
    )
    @pytest.mark.parametrize(
        "attribute_type2,attribute_default2,other_value2", properties
    )
    def test_with_two_other_property(
        self,
        attribute_type1,
        attribute_default1,
        other_value1,
        attribute_type2,
        attribute_default2,
        other_value2,
        kwarg_type,
        values,
    ):

        class MyObject(PrintConstructionCallMixin):
            def __init__(
                self,
                other1: attribute_type1 = attribute_default1,
                other2: attribute_type2 = attribute_default2,
                **a: kwarg_type,
            ):
                self.other1 = other1
                self.other2 = other2
                self.a = a

        assert (
            str(MyObject(other_value1, other_value2, **values))
            == f"MyObject(other1={other_value1},other2={other_value2},{','.join(map(lambda k: f'{k}={values[k]}', values))})"
        )
        assert (
            str(MyObject(other_value1, other_value2))
            == f"MyObject(other1={other_value1},other2={other_value2})"
        )


@pytest.mark.parametrize("arg_type,arg_values", arg_properties)
@pytest.mark.parametrize("kwarg_type,kwarg_values", kwarg_properties)
class TestArgAndKwargParameters:

    def test(self, arg_type, arg_values, kwarg_type, kwarg_values):
        class MyObject(PrintConstructionCallMixin):
            def __init__(self, *a: arg_type, **b: kwarg_type):
                self.a = a
                self.b = b

        assert (
            str(MyObject(*arg_values, **kwarg_values))
            == f"MyObject({','.join(map(str, arg_values))},{','.join(map(lambda k: f'{k}={kwarg_values[k]}', kwarg_values))})"
        )
        assert (
            str(MyObject(*arg_values)) == f"MyObject({','.join(map(str, arg_values))})"
        )
        assert (
            str(MyObject(**kwarg_values))
            == f"MyObject({','.join(map(lambda k: f'{k}={kwarg_values[k]}', kwarg_values))})"
        )

    @pytest.mark.parametrize("attribute_type,attribute_default,other_value", properties)
    def test_with_other_property(
        self,
        attribute_type,
        attribute_default,
        other_value,
        arg_type,
        arg_values,
        kwarg_type,
        kwarg_values,
    ):

        class MyObject(PrintConstructionCallMixin):
            def __init__(
                self,
                other: attribute_type = attribute_default,
                *a: arg_type,
                **b: kwarg_type,
            ):
                self.other = other
                self.a = a
                self.b = b

        assert (
            str(MyObject(other_value, *arg_values, **kwarg_values))
            == f"MyObject({other_value},{','.join(map(str, arg_values))},{','.join(map(lambda k: f'{k}={kwarg_values[k]}', kwarg_values))})"
        )
        assert (
            str(MyObject(other_value, *arg_values))
            == f"MyObject({other_value},{','.join(map(str, arg_values))})"
        )
        assert (
            str(MyObject(other_value, **kwarg_values))
            == f"MyObject(other={other_value},{','.join(map(lambda k: f'{k}={kwarg_values[k]}', kwarg_values))})"
        )
        assert (
            str(MyObject(**kwarg_values))
            == f"MyObject({','.join(map(lambda k: f'{k}={kwarg_values[k]}', kwarg_values))})"
        )
