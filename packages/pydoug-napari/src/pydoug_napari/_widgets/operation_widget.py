"""
Convert PyDoug core functions into napari widgets.
"""

import inspect

from magicclass.widgets import CollapsibleContainer
from magicgui import magicgui
from magicgui.widgets import FunctionGui
from napari.layers import Image
from napari.qt.threading import thread_worker
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

        @thread_worker
        def _run():
            """
            Run the operation widget's function and return its results.
            """

            return operation_spec.func(layer.data, **params)  # pragma: no cover
            # covered in a separate Qt threading event

        def _on_done(result):
            """
            Calls back on main Qt thread to return result after operation is performed.

            Parameters
            ----------
            result
                The result of the operation.
            """

            result_layer: Image | list[Image] = viewer.add_image(
                result, name=f"{operation_spec.display_name}"
            )
            recorder.record(operation_spec.name, params, layer_ref=result_layer)

        worker = _run()  # type: ignore[call-arg]
        worker.returned.connect(_on_done)
        worker.start()

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
        **operation_spec.widget_options,
    )
    for field_name, native_calls in operation_spec.native_options.items():
        field_widget = getattr(widget, field_name)
        for method_name, arg in native_calls.items():
            getattr(field_widget.native, method_name)(arg)
    connect_layer_autoselect(viewer, widget, "layer")
    container: CollapsibleContainer = CollapsibleContainer(
        widgets=[widget], labels=False
    )
    container.text = operation_spec.display_name

    return container
