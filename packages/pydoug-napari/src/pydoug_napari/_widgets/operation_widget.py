"""
Convert PyDoug core functions into napari widgets.
"""

import inspect

from magicclass.widgets import CollapsibleContainer
from magicgui import magicgui
from magicgui.widgets import FunctionGui
from napari.layers import Image
from napari.viewer import Viewer

from pydoug_core.batch.recorder import Recorder
from pydoug_core.registry import OperationSpec
from pydoug_napari._widgets._layer_sync import connect_layer_autoselect


def make_operation_widget(
    operation_spec: OperationSpec, viewer: Viewer, recorder: Recorder
) -> CollapsibleContainer:
    """
    Automatically generate widgets from the PyDoug core registry and allow their use
    to be recorded in the parameters log.

    Parameters
    ----------
    operation_spec: OperationSpec
        The operation and its specifications.
    viewer: Viewer
        The napari viewer object.
    recorder: Recorder
        The recorder object for the parameters log.

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
        recorder.record(operation_spec.name, params)
        viewer.add_image(result, name=f"{operation_spec.display_name}")

    params_only = [
        p.replace(kind=inspect.Parameter.KEYWORD_ONLY)
        for p in list(operation_spec.signature.parameters.values())[1:]
    ]
    _dispatch.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [
            inspect.Parameter(
                "layer", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Image
            )
        ]
        + params_only
    )
    widget: FunctionGui = magicgui(
        _dispatch,
        call_button=f"Run {operation_spec.display_name}",
        layer={"label": "Layer"},
    )
    connect_layer_autoselect(viewer, widget, "layer")
    container: CollapsibleContainer = CollapsibleContainer(
        widgets=[widget], labels=False
    )
    container.text = operation_spec.display_name

    return container
