"""
Convert PyDoug core functions into napari widgets.
"""

import inspect
from typing import Any
from uuid import uuid4

import numpy as np
from docstring_parser import parse as parse_docstring
from magicclass.widgets import CollapsibleContainer
from magicgui import magicgui
from magicgui.widgets import FunctionGui
from matplotlib.figure import Figure
from napari.layers import Image
from napari.qt.threading import thread_worker
from napari.viewer import Viewer

from pydoug_core.batch.recorder import Recorder
from pydoug_core.registry import OperationSpec
from pydoug_napari._widgets._layer_sync import connect_layer_autoselect
from pydoug_napari._widgets._result_display import default_display_result


def make_operation_widget(
    operation_spec: OperationSpec, viewer: Viewer, recorder: Recorder
) -> CollapsibleContainer:
    """
    Automatically generate widgets from the core registry and allow their use to be
    recorded in the parameters log.

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
    """

    mask_names: set[str] = set(operation_spec.mask_params)
    is_running: dict[str, bool] = {"value": False}

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

        if is_running["value"]:
            return

        is_running["value"] = True
        if widget.call_button is not None:
            widget.call_button.native.setEnabled(False)
            widget.call_button.text = "Running..."

        call_params: dict = {}
        recorded_params: dict = {}
        input_refs: dict[str, str] = {}
        pending_masks: dict[str, np.ndarray] = {}

        for name, value in params.items():
            if name in mask_names:
                if value is not None:
                    mask_layer: Image = value
                    call_params[name] = mask_layer.data
                    step_id: str | None = recorder.step_id_for_layer(mask_layer)
                    if step_id is not None:
                        input_refs[name] = f"step:{step_id}"
                    else:
                        input_refs[name] = f"pending:{uuid4().hex[:8]}"
                        if isinstance(mask_layer.data, np.ndarray):
                            pending_masks[name] = np.copy(mask_layer.data)
                else:
                    call_params[name] = value
                    recorded_params[name] = value
            else:
                call_params[name] = value
                recorded_params[name] = value

        layer_step_id: str | None = recorder.step_id_for_layer(layer)
        input_refs["layer"] = f"step:{layer_step_id}" if layer_step_id else "primary"

        @thread_worker(ignore_errors=True)
        def _run() -> Any:
            """
            Run the operation widget's function and return its results.
            """

            return operation_spec.func(layer.data, **call_params)  # pragma: no cover
            # covered in a separate Qt threading event

        def _reset_button() -> None:
            """
            Reset the widget's call button once the operation is complete.
            """

            is_running["value"] = False
            if widget.call_button is not None:
                widget.call_button.native.setEnabled(True)
                widget.call_button.text = f"Run {operation_spec.display_name}"

        def _on_done(
            result: dict[str, float | int] | Figure | float | int | np.ndarray | str,
        ) -> None:
            """
            Calls back on main Qt thread to return result after operation is performed.

            Parameters
            ----------
            result: np.ndarray
                The result layer of the operation.
            """

            _reset_button()
            if operation_spec.output_kind == "image":
                result_layer: Image | list[Image] = viewer.add_image(
                    result, name=f"{operation_spec.display_name}"
                )
                recorder.record(
                    operation_spec.name,
                    recorded_params,
                    input_refs=input_refs,
                    layer_ref=result_layer,
                    pending_masks=pending_masks,
                )
            else:
                assert not isinstance(result, np.ndarray)
                (operation_spec.display_result or default_display_result)(result)
                recorder.record(
                    operation_spec.name,
                    recorded_params,
                    input_refs=input_refs,
                    layer_ref=None,
                    pending_masks=pending_masks,
                )

        def _on_error(exc: Exception) -> None:
            """
            Raise exception and reset call button when an error is encountered.
            """

            _reset_button()
            raise exc

        worker = _run()  # type: ignore[call-arg]
        worker.returned.connect(_on_done)
        worker.errored.connect(_on_error)
        worker.start()

    params_only: list = []
    for p in list(operation_spec.signature.parameters.values())[1:]:
        if p.name in mask_names:
            p = p.replace(kind=inspect.Parameter.KEYWORD_ONLY, annotation=Image)
        else:
            p = p.replace(kind=inspect.Parameter.KEYWORD_ONLY)
        params_only.append(p)

    _dispatch.__doc__ = operation_spec.func.__doc__
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
        layer={"label": "Layer", "tooltip": "The image layer to process."},
        **operation_spec.widget_options,
    )

    for field_name, native_calls in operation_spec.native_options.items():
        field_widget = getattr(widget, field_name)
        for method_name, arg in native_calls.items():
            getattr(field_widget.native, method_name)(arg)

    connect_layer_autoselect(viewer, widget, "layer", autoselect=True)
    for mask_name in mask_names:
        connect_layer_autoselect(viewer, widget, mask_name, autoselect=False)

    container: CollapsibleContainer = CollapsibleContainer(
        widgets=[widget], labels=False
    )
    container.text = operation_spec.display_name
    container.native.setObjectName("Collapsible")
    container.native.setStyleSheet(
        "QWidget#Collapsible { border: 1px solid #888; border-radius: 4px; }"
    )
    container.native.setContentsMargins(2, 2, 2, 6)

    parsed_doc = parse_docstring(operation_spec.func.__doc__ or "")
    summary = " ".join(
        filter(None, [parsed_doc.short_description, parsed_doc.long_description])
    )
    if summary:
        if widget.call_button is not None:
            widget.call_button.tooltip = summary
        container.native.setToolTip(summary)

    return container
