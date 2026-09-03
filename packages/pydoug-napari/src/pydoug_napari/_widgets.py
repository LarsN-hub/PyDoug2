"""
napari plugin for PyDoug's core functions as widgets.
"""

import inspect
from collections.abc import Callable

from magicgui import magicgui
from napari.layers import Image
from napari.viewer import Viewer

from pydoug_core.batch.recorder import Recorder
from pydoug_core.registry import OperationSpec


def make_operation_widget(operation_spec: OperationSpec, viewer: Viewer) -> Callable:
    """
    Automatically generate widgets from the PyDoug core registry.

    Parameters
    ----------
    operation_spec: OperationSpec
        The operation and its specifications.
    viewer: Viewer
        The napari viewer object.

    Returns
    -------
    funcgui: Callable
        A magicgui widget.
    -------

    """

    def _dispatch(layer: Image, **params) -> None:
        """
        Record an operation in the parameters log.

        Parameters
        ----------
        layer: Image
            napari Image layer
        **params
            The parameters for the function.
            The keyword arguments are passed to "Recorder.record()."
        """

        result = operation_spec.func(layer.data, **params)
        Recorder.record(operation_spec.name, params)
        viewer.add_image(result, name=f"{layer.name}_{operation_spec.name}")

    params_only = [
        p
        for name, p in operation_spec.signature.parameters.items()
        if name != next(iter(operation_spec.signature.parameters))
    ]
    _dispatch.__signature__ = inspect.Signature(
        [
            inspect.Parameter(
                "layer", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Image
            )
        ]
        + params_only
    )

    return magicgui(_dispatch, call_button=f"Run {operation_spec.display_name}")
