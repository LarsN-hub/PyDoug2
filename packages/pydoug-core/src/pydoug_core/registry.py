"""
Registry framework for core functions.
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class OperationSpec:
    """
    Define specifications for an operation to be registered.

    Attributes
    ----------
    name: str
        The name of the operation
    func: Callable
        The function called by the operation.
    category: str
        The category of the function/operation.
    display_name: str
        The name to be displayed for the operation.
    widget_options: dict
        Options to modify default widget parameters.
    native_options: dict[str, dict]
        Options to modify the default widget's native parameters.
    signature: inspect.Signature
        The function's signature.
    """

    name: str
    func: Callable
    category: str
    display_name: str
    widget_options: dict = field(default_factory=dict)
    native_options: dict[str, dict] = field(default_factory=dict)
    signature: inspect.Signature = field(init=False)

    def __post_init__(self):
        self.signature = inspect.signature(self.func)


_REGISTRY: dict[str, OperationSpec] = {}


def register_operation(
    category: str,
    display_name: None | str = None,
    widget_options: dict | None = None,
    native_options: dict[str, dict] | None = None,
) -> Callable:
    """
    Provide a decorator to register a function as an operation.

    Parameters
    ----------
    category: str
        The category of the operation.
    display_name: None | str = None
        The display name of the operation (optional).
        Automatically generated if not provided.
    widget_options: dict | None
        (Optional) Dictionary of default widget parameter settings.
    native_options: dict[str, dict] | None
        (Optional) Dictionary of default widget native parameter settings.

    Returns
    -------
    decorator: Callable
        A function to register the provided function as an operation.
    """

    def decorator(func: Callable) -> Callable:
        """
        Register a function as a discoverable, recordable operation.

        Parameters
        ----------
        func: Callable
            The function to be registered as an operation.

        Returns
        -------
        func: Callable
            The function to be registered as an operation.
        """

        _REGISTRY[func.__name__] = OperationSpec(
            name=func.__name__,
            func=func,
            category=category,
            display_name=display_name or func.__name__.replace("_", " ").title(),
            widget_options=widget_options or {},
            native_options=native_options or {},
        )
        return func

    return decorator


def unregister_operation(name: str) -> None:
    """
    Given an operation's name, unregister it.

    Parameters
    ----------
    name: str
        The operation's name.
    """

    _REGISTRY.pop(name, None)


def get_operation(name: str) -> OperationSpec:
    """
    Retrieve an operation with its name.

    Parameters
    ----------
    name: str
        The name of the operation.

    Returns
    -------
    opspec: OperationSpec
        The retrieved operation.
    """

    return _REGISTRY[name]


def list_operations(category: None | str = None) -> list[OperationSpec]:
    """
    Return a list of operations in the registry. Can specify a specific category.

    Parameters
    ----------
    category: None | str = None
        The category of the operation to be returned (optional).

    Returns
    -------
    opspec_list: list[OperationSpec]
        The list of operations in the registry.
    """

    ops = _REGISTRY.values()
    return [o for o in ops if category is None or o.category == category]
