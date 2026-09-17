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
from napari.layers import Image, Layer
from napari.qt.threading import thread_worker
from napari.viewer import Viewer

from pydoug_core.batch.recorder import Recorder
from pydoug_core.registry import OperationSpec
from pydoug_napari._widgets._layer_sync import connect_layer_autoselect
from pydoug_napari._widgets.plot_display_widget import default_display_plot
from pydoug_napari._widgets.results_log_widget import ResultsLog


def make_operation_widget(
    operation_spec: OperationSpec,
    viewer: Viewer,
    recorder: Recorder,
    results_log: ResultsLog,
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
        The recorder object for the workflow.
    results_log: ResultsLog
        The results log object for non-image results.

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

        record_in_workflow: bool = params.pop("Record_in_Workflow", True)
        live_preview: bool = params.pop("Preview", False)

        call_params: dict = {}
        recorded_params: dict = {}
        input_refs: dict[str, str] = {}
        pending_masks: dict[str, np.ndarray] = {}
        mask_enabled: bool = params.get("Apply_Mask", True)

        for name, value in params.items():
            if name in mask_names:
                if mask_enabled and value is not None:
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
                    call_params[name] = None
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
            result: dict[str, float | int | str]
            | Figure
            | float
            | int
            | np.ndarray
            | str,
        ) -> None:
            """
            Calls back on main Qt thread to return result after operation is performed.

            Parameters
            ----------
            result: dict[str, float | int | str] | Figure | float | int | np.ndarray
            | str
                The result layer of the operation.
            results_log: ResultsLog
                The results log object.
            """

            _reset_button()

            should_record: bool = operation_spec.recordable and (
                record_in_workflow if operation_spec.recordable_toggle else True
            )

            if live_preview and preview_layer_ref["layer"] is not None:
                viewer.layers.remove(preview_layer_ref["layer"])
                preview_layer_ref["layer"] = None
                widget.Preview.value = False

            if operation_spec.output_kind == "image":
                in_place: bool = operation_spec.allow_in_place and call_params.get(
                    "In_Place", False
                )
                if in_place:
                    layer.data = result  # type: ignore
                    result_layer = layer
                else:
                    result_layer: Image | list[Image] = viewer.add_image(
                        result, name=f"{operation_spec.display_name}"
                    )
                if should_record:
                    recorder.record(
                        operation_spec.name,
                        recorded_params,
                        input_refs=input_refs,
                        layer_ref=result_layer,
                        pending_masks=pending_masks,
                    )

            elif operation_spec.output_kind == "plot":
                (operation_spec.display_result or default_display_plot)(result)  # type: ignore
                results_log.add(
                    operation_spec.display_name, "Plot generated, see window"
                )
                if record_in_workflow and should_record:
                    recorder.record(
                        operation_spec.name,
                        recorded_params,
                        input_refs=input_refs,
                        layer_ref=None,
                        pending_masks=pending_masks,
                    )

            else:  # "value"
                results_log.add(operation_spec.display_name, result)  # type: ignore
                if record_in_workflow and should_record:
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

    extra_params: list = []
    if operation_spec.recordable_toggle:
        extra_params.append(
            inspect.Parameter(
                "Record_in_Workflow",
                inspect.Parameter.KEYWORD_ONLY,
                annotation=bool,
                default=False,
            )
        )
    if operation_spec.live_preview:
        extra_params.append(
            inspect.Parameter(
                "Preview",
                inspect.Parameter.KEYWORD_ONLY,
                annotation=bool,
                default=False,
            )
        )

    _dispatch.__doc__ = operation_spec.func.__doc__
    _dispatch.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [
            inspect.Parameter(
                "layer", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Image
            )
        ]
        + params_only
        + extra_params
    )

    extra_widget_options: dict[str, dict[str, str]] = {}
    if operation_spec.recordable_toggle:
        extra_widget_options["Record_in_Workflow"] = {
            "tooltip": "Uncheck to run this without saving it to the workflow."
        }
    if operation_spec.live_preview:
        extra_widget_options["Preview"] = {
            "tooltip": "Check to see a preview of the thresholding segmentation."
        }

    widget: FunctionGui = magicgui(
        _dispatch,
        call_button=f"Run {operation_spec.display_name}",
        layer={"label": "Layer", "tooltip": "The image layer to process."},
        **extra_widget_options,
        **operation_spec.widget_options,
    )

    if operation_spec.allow_in_place:
        widget.In_Place.tooltip = "Warning: overwrites the selected layer's data "
        "directly. This is much faster and conserves memory, but cannot be undone."

    if operation_spec.live_preview:
        preview_layer_ref: dict[str, Layer | None] = {"layer": None}

        def _current_real_params() -> dict:
            real_names = [p.name for p in operation_spec.signature.parameters.values()][
                1:
            ]
            return {
                name: getattr(widget, name).value
                for name in real_names
                if hasattr(widget, name)
            }

        def _update_preview(event=None):
            if not widget.Preview.value or widget.layer.value is None:
                return
            result = operation_spec.func(
                widget.layer.value.data, **_current_real_params()
            )
            if preview_layer_ref["layer"] is None:
                preview_layer_ref["layer"] = viewer.add_labels(  # type: ignore
                    result, name=f"{operation_spec.display_name} (preview)", opacity=0.5
                )
            else:
                preview_layer_ref["layer"].data = result

        def _on_preview_toggled(event):
            if widget.Preview.value:
                _update_preview()
            elif preview_layer_ref["layer"] is not None:
                viewer.layers.remove(preview_layer_ref["layer"])
                preview_layer_ref["layer"] = None

        widget.changed.connect(_update_preview)
        widget.Preview.changed.connect(_on_preview_toggled)

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
