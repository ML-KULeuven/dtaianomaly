import pytest
from sklearn.exceptions import NotFittedError

from dtaianomaly.utils import CheckIsFittedMixin


class ObjectRequiresFittingWithProperties(CheckIsFittedMixin):
    a: None
    b: None
    c_: None

    def __init__(self):
        self.a, self.b = None, None

    def fit(self):
        self.c_ = None


class ObjectRequiresFittingNoProperties(CheckIsFittedMixin):
    c_: None

    def fit(self):
        self.c_ = None


class ObjectRequiresNoFittingWithProperties(CheckIsFittedMixin):
    a: None
    b: None

    def __init__(self):
        self.a, self.b = None, None

    def fit(self):
        pass


class ObjectRequiresNoFittingNoProperties(CheckIsFittedMixin):
    def fit(self):
        pass


def asserts(obj: CheckIsFittedMixin, requires_fitting: bool, is_fitted: bool):
    if requires_fitting:
        assert obj.requires_fitting()
        if is_fitted:
            assert obj.is_fitted()
            obj.check_is_fitted()
        else:
            assert not obj.is_fitted()
            with pytest.raises(NotFittedError):
                obj.check_is_fitted()

    else:
        assert not obj.requires_fitting()
        assert obj.is_fitted()
        obj.check_is_fitted()


@pytest.mark.parametrize(
    "cls,requires_fitting",
    [
        (ObjectRequiresFittingWithProperties, True),
        (ObjectRequiresFittingNoProperties, True),
        (ObjectRequiresNoFittingWithProperties, False),
        (ObjectRequiresNoFittingNoProperties, False),
    ],
)
class TestCheckIsFittedMixinRequiresFitting:

    def test(self, cls, requires_fitting):
        instance = cls()
        asserts(instance, requires_fitting, False)
        instance.fit()
        asserts(instance, requires_fitting, True)

    def test_child_with_fitting(self, cls, requires_fitting):
        class Child(cls):
            child_attribute_: int

            def fit(self):
                super().fit()
                self.child_attribute_ = 1

        child = Child()
        asserts(child, True, False)
        child.fit()
        asserts(child, True, True)

    def test_child_without_fitting(self, cls, requires_fitting):
        class Child(cls):
            pass

        child = Child()
        asserts(child, requires_fitting, False)
        child.fit()
        asserts(child, requires_fitting, True)
