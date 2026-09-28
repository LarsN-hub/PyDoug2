"""
Registry framework for core functions.
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


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
    array_params: tuple[str, ...]
        Image layer parameters as required.
    array_param_types: dict[str, str]
        The layer type for each array input.
    array_param_flags: dict[str, str]
        Relevant flags for array inputs.
    output_kind: Literal["image", "labels", "plot", "value"]
        Whether the operation should output an image or a value.
    display_result: Callable[[Any], None] | None
        How to interactively display the result.
    save_result: Callable[[Any, Path], None] | None
        How to save a non-image result.
    recordable_toggle: bool
        Whether or not to add a checkbox to the widget to toggle recording.
    allow_in_place: bool
        Whether or not to add a checkbox to the widget to toggle in-place operations.
    recordable: bool
        Whether or not the operation should be recorded on use.
    live_preview: bool
        Whether or not to allow a live preview of the operation.
    layer_type: Literal["image", "labels"]
        The layer type to be added if a numpy array is produced.
    captures_viewer_settings: bool = False
        Whether or not the operation should capture viewer sttings (for screenshots).
    signature: inspect.Signature
        The function's signature.
    """

    name: str
    func: Callable
    category: str
    display_name: str
    widget_options: dict = field(default_factory=dict)
    native_options: dict[str, dict] = field(default_factory=dict)
    array_params: tuple[str, ...] = ()
    array_param_types: dict[str, str] = field(default_factory=dict)
    array_param_flags: dict[str, str] = field(default_factory=dict)
    output_kind: Literal["image", "labels", "plot", "value"] = "image"
    display_result: Callable[[Any], None] | None = None
    save_result: Callable[[Any, Path], None] | None = None
    recordable_toggle: bool = False
    allow_in_place: bool = False
    recordable: bool = True
    live_preview: bool = False
    layer_type: Literal["image", "labels"] = "image"
    captures_viewer_settings: bool = False
    signature: inspect.Signature = field(init=False)

    def __post_init__(self):
        self.signature = inspect.signature(self.func)


_REGISTRY: dict[str, OperationSpec] = {}


def register_operation(
    category: str,
    display_name: None | str = None,
    widget_options: dict | None = None,
    native_options: dict[str, dict] | None = None,
    array_params: tuple[str, ...] = (),
    array_param_types: dict[str, str] = {},
    array_param_flags: dict[str, str] = {},
    output_kind: Literal["image", "labels", "plot", "value"] = "image",
    display_result: Callable[[Any], None] | None = None,
    save_result: Callable[[Any, Path], None] | None = None,
    recordable_toggle: bool = False,
    recordable: bool = True,
    live_preview: bool = False,
    layer_type: Literal["image", "labels"] = "image",
    captures_viewer_settings: bool = False,
    allow_in_place: bool = False,
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
    array_params: tuple[str, ...]
        (Optional) Tuple of array parameters as needed for the operation.
    array_param_types: dict[str, str]
        The napari layer type for each array input.
    array_param_flags: dict[str, str]
        Relevant flags for array inputs.
    output_kind: Literal["image", "labels", "plot", "value"]
        Whether the operation should output an image or a value.
    display_result: Callable[[Any], None] | None
        How to interactively display the result.
    save_result: Callable[[Any, Path], None] | None
        How to save a non-image result.
    recordable_toggle: bool
        Whether or not to add a checkbox to the widget to toggle recording.
    recordable: bool
        Whether or not the operation should be recorded on use.
    allow_in_place: bool
        Whether or not to add a checkbox to the widget to toggle in-place operations.
    live_preview: bool
        Whether or not to allow a live preview of the operation.
    layer_type: Literal["image", "labels"]
        The layer type to be added if a numpy array is produced.
    captures_viewer_settings: bool = False
        Whether or not the operation should capture viewer sttings (for screenshots).

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

        if recordable_toggle and output_kind == "image":
            raise ValueError(  # pragma: no cover
                f"{func.__name__}: recordable_toggle only makes sense for 'value'/"
                "'plot' operations: skipping the recording of an image step would leave"
                " later steps referencing its layer with a dangling input reference."
            )

        sig: inspect.Signature = inspect.signature(func)
        if allow_in_place and "In_Place" not in sig.parameters:
            raise ValueError(  # pragma: no cover
                f"{func.__name__}: allow_in_place=True requires the function to "
                "declare its own 'In_Place: bool' parameter — only the operation "
                "itself can correctly implement zero-copy behavior."
            )

        if allow_in_place and output_kind not in ("image",):
            raise ValueError(  # pragma: no cover
                f"{func.__name__}: allow_in_place only makes sense for "
                "output_kind='image'."
            )

        if not recordable and output_kind == "image":
            raise ValueError(  # pragma: no cover
                f"{func.__name__}: recordable=False on an 'image' operation means its "
                "output layer is never tracked, so any later step selecting that layer "
                "would silently fall back to referencing the original input instead — "
                "breaking batch replay. Only use recordable=False for output_kind that "
                "isn't consumed downstream as an image/mask input."
            )

        valid: set[str] = {"image", "labels"}
        if layer_type not in valid:
            raise ValueError(  # pragma: no cover
                f"{func.__name__}: layer_type must be one of "
                f"{valid}, got {layer_type!r}"
            )
        bad: set[str] = set((array_param_types or {}).values()) - valid
        if bad:
            raise ValueError(  # pragma: no cover
                f"{func.__name__}: array_param_types values must be one of "
                f"{valid}, got {bad}"
            )

        _REGISTRY[func.__name__] = OperationSpec(
            name=func.__name__,
            func=func,
            category=category,
            display_name=display_name or func.__name__.replace("_", " ").title(),
            widget_options=widget_options or {},
            native_options=native_options or {},
            array_params=tuple(array_params),
            array_param_types=array_param_types or {},
            array_param_flags=array_param_flags or {},
            output_kind=output_kind,
            display_result=display_result,
            save_result=save_result,
            recordable_toggle=recordable_toggle,
            allow_in_place=allow_in_place,
            recordable=recordable,
            live_preview=live_preview,
            layer_type=layer_type,
            captures_viewer_settings=captures_viewer_settings,
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
