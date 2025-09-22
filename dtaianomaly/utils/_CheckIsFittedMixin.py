import abc
from collections import ChainMap

from sklearn.exceptions import NotFittedError

__all__ = ["CheckIsFittedMixin"]


class CheckIsFittedMixin(abc.ABC):
    """
    Mixin object to automatically check whether the object is fitted.

    Mixin object to add functionality for automatically checking whether
    an object has been fitted or not. Deciding if an object is fitted is
    based on the convention that attributes to fit end with an underscore.
    Using this convention, this class checks whether all attributes ending
    with an underscore have been initialized or not. It also checks if any
    such attribute exists to decide if the object must be fitted.

    Examples
    --------
    >>> from dtaianomaly.utils import CheckIsFittedMixin
    >>>
    >>> class MyObject(CheckIsFittedMixin):
    ...     parameter: int
    ...     attribute_to_fit_: int
    ...
    ...     def __init__(self, positive_int: int):
    ...         self.positive_int = positive_int
    ...
    ...     def fit(self):
    ...         self.attribute_to_fit_ = 0
    >>>
    >>> obj = MyObject(5)
    >>> obj.requires_fitting()
    True
    >>> obj.is_fitted()
    False
    >>> obj.fit()
    >>> obj.requires_fitting()
    True
    >>> obj.is_fitted()
    True
    """

    def requires_fitting(self) -> bool:
        """
        Checks whether this object requires fitting.

        Checks whether any of the attributes of this object ends with an
        underscore ('_'), which indicates that the attribute is set when
        the object is fitted. Note that this method does not check whether
        the object is fitted, i.e., whether the attributes have been set.

        Returns
        -------
        requires_fitting: bool
            True if and only if this object has attributes that end with '_'.
        """
        return any(attr.endswith("_") for attr in self._all_annotations())

    def is_fitted(self) -> bool:
        """
        Checks whether this object is fitted.

        Checks whether all the attributes of this object that end with
        an underscore ('_') has been initialized.

        Returns
        -------
        is_fitted: bool
            True if and only if all the attributes of this object ending
            with '_' are initialized.
        """
        return all(
            hasattr(self, attr)
            for attr in self._all_annotations()
            if attr.endswith("_")
        )

    def check_is_fitted(self) -> None:
        """
        Raise an error if this object is not fitted.

        Checks whether this object is fitted, and if it is not fitted,
        an exception is thrown.

        Raises
        ------
        NotFittedError
            If this object is not fitted.
        """
        if not self.is_fitted():
            raise NotFittedError(
                f"This '{self.__class__.__name__}')' has not been fitted yet!"
            )

    def _all_annotations(self) -> ChainMap:
        """
        Returns a dictionary-like ChainMap that includes annotations for all
        attributes defined in cls or inherited from superclasses.
        """
        return ChainMap(
            *(
                c.__annotations__
                for c in type(self).__mro__
                if "__annotations__" in c.__dict__
            )
        )
