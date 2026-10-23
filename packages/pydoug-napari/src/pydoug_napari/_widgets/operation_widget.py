"""
Convert PyDoug core functions into napari widgets.
"""

import inspect
from typing import Any

import numpy as np
from docstring_parser import parse as parse_docstring
from magicclass.widgets import CollapsibleContainer
from magicgui import magicgui
from magicgui.widgets import FunctionGui
from matplotlib.figure import Figure
from napari.layers import Labels, Layer
from napari.qt.threading import thread_worker
from napari.viewer import Viewer

from pydoug_core.batch.recorder import Recorder
from pydoug_core.registry import OperationSpec
from pydoug_core.results import PlotResult, ValueResult, VisualResult
from pydoug_napari._widgets._layer_data import LAYER_TYPE_MAP, layer_payload
from pydoug_napari._widgets._layer_sync import connect_layer_autoselect
from pydoug_napari._widgets._settings_capture import to_jsonable
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

    array_names: set[str] = set(operation_spec.array_params)
    is_running: dict[str, bool] = {"value": False}

    def _dispatch(**params) -> None:
        """
        Record an operation in the parameters log.

        Parameters
        ----------
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

        layer = params.pop("layer", None) if operation_spec.requires_layer else None
        record_in_workflow: bool = params.pop("Record_in_Workflow", True)
        live_preview: bool = params.pop("Preview", False)

        call_params: dict = {}
        recorded_params: dict = {}
        input_refs: dict[str, str] = {}

        primary_payload: Any = (
            layer_payload(layer, live_shapes=not operation_spec.recordable)
            if operation_spec.requires_layer
            else None
        )

        for name, value in params.items():
            if name in array_names:
                flag_name: str | None = operation_spec.array_param_flags.get(name)
                enabled: bool = params.get(flag_name, True) if flag_name else True
                if enabled and value is not None:
                    array_layer: Any = value
                    payload: Any = layer_payload(
                        array_layer, live_shapes=not operation_spec.recordable
                    )
                    call_params[name] = payload
                    if operation_spec.recordable:
                        step_id: str | None = recorder.step_id_for_layer(array_layer)
                        if step_id is not None:
                            input_refs[name] = f"step:{step_id}"
                        elif name in operation_spec.primary_params:
                            input_refs[name] = "primary"
                        else:
                            input_refs[name] = recorder.get_or_create_array_ref(
                                array_layer, payload
                            )
                else:
                    call_params[name] = None
            else:
                call_params[name] = value
                recorded_params[name] = value

        if operation_spec.captures_viewer_settings:
            call_params["Settings"] = dict(captured_settings)
            recorded_params["Settings"] = dict(captured_settings)

        if operation_spec.requires_layer and operation_spec.recordable:
            layer_step_id = recorder.step_id_for_layer(layer)
            if layer_step_id is not None:
                input_refs["layer"] = f"step:{layer_step_id}"
            elif operation_spec.layer_type in ("image", "labels", "visual"):
                input_refs["layer"] = "primary"
            else:  # surface or shapes
                input_refs["layer"] = recorder.get_or_create_array_ref(
                    layer, primary_payload
                )

        def _reset_button() -> None:
            """
            Reset the widget's call button once the operation is complete.
            """

            is_running["value"] = False
            if widget.call_button is not None:
                widget.call_button.native.setEnabled(True)
                widget.call_button.text = f"Run {operation_spec.display_name}"

        def _on_done(
            result: dict
            | Figure
            | float
            | int
            | np.ndarray
            | str
            | PlotResult
            | ValueResult
            | VisualResult,
        ) -> None:
            """
            Calls back on main Qt thread to return result after operation is performed.

            Parameters
            ----------
            result: dict | Figure | float | int | np.ndarray | str | PlotResult |
            ValueResult | VisualResult
                The result of the operation.
            """

            _reset_button()

            should_record: bool = operation_spec.recordable and (
                record_in_workflow if operation_spec.recordable_toggle else True
            )

            if live_preview and preview_layer_ref["layer"] is not None:
                viewer.layers.remove(preview_layer_ref["layer"])
                preview_layer_ref["layer"] = None
                widget.Preview.value = False

            if operation_spec.output_kind in ("image", "labels", "surface"):
                in_place: bool = (
                    operation_spec.requires_layer
                    and operation_spec.allow_in_place
                    and call_params.get("In_Place", False)
                )
                if in_place:
                    layer.data = result  # type: ignore
                    result_layer = layer
                else:
                    add_fn = {
                        "image": viewer.add_image,
                        "labels": viewer.add_labels,  # type: ignore
                        "surface": viewer.add_surface,  # type: ignore
                    }[operation_spec.output_kind]
                    result_layer: Any = add_fn(
                        result, name=f"{operation_spec.display_name}"
                    )
                if should_record:
                    recorder.record(
                        operation_spec.name,
                        recorded_params,
                        input_refs=input_refs,
                        layer_ref=result_layer,
                    )

            elif operation_spec.output_kind == "visual":
                add_fn = {
                    "image": viewer.add_image,
                    "labels": viewer.add_labels,  # type: ignore
                }[result.layer_type]  # type: ignore
                result_layer: Any = add_fn(
                    result.imarray,  # type: ignore
                    name=f"{operation_spec.display_name}",
                    **result.vis_dict,  # type: ignore
                )
                if should_record:
                    recorder.record(
                        operation_spec.name,
                        recorded_params,
                        input_refs=input_refs,
                        layer_ref=result_layer,
                    )

            else:  # "value", "plot"
                step_id: str | None = None
                if record_in_workflow and should_record:
                    step_id = recorder.record(
                        operation_spec.name,
                        recorded_params,
                        input_refs=input_refs,
                        layer_ref=None,
                    )
                name: str = operation_spec.display_name
                dataset: str = (
                    layer.name
                    if operation_spec.requires_layer and layer is not None
                    else name
                )
                if isinstance(result, (PlotResult, Figure)):
                    figure: Figure = (
                        result.figure if isinstance(result, PlotResult) else result
                    )
                    (operation_spec.display_result or default_display_plot)(figure)
                    results_log.add(
                        name,
                        "Plot generated, see window",
                        result if isinstance(result, PlotResult) else None,
                        dataset,
                        operation_spec.units,
                        step_id,
                    )
                    if isinstance(result, PlotResult):
                        for table_name, table in result.tables.items():
                            results_log.add(
                                f"{operation_spec.display_name} ({table_name})",
                                ValueResult(table, result.units).display(),
                                step_id=step_id,
                            )
                else:
                    results_log.add(
                        name,
                        result.display() if isinstance(result, ValueResult) else result,
                        result,
                        dataset,
                        operation_spec.units,
                        step_id,
                    )

        def _on_error(exc: Exception) -> None:
            """
            Raise exception and reset call button when an error is encountered.

            Parameters
            ----------
            exc: Exception
                The error that ocurred.
            """

            _reset_button()
            raise exc

        @thread_worker(ignore_errors=True)
        def _run() -> Any:  # pragma: no cover
            """
            Run the operation widget's function and return its results.

            Returns
            -------
            output: Any
                The output of the operation's function.
            """

            if operation_spec.requires_layer:
                return operation_spec.func(primary_payload, **call_params)  # type: ignore
            return operation_spec.func(**call_params)

        worker = _run()  # type: ignore
        worker.returned.connect(_on_done)
        worker.errored.connect(_on_error)
        worker.start()

    start_index: int = 1 if operation_spec.requires_layer else 0
    params_only: list = []
    for p in list(operation_spec.signature.parameters.values())[start_index:]:
        if operation_spec.captures_viewer_settings and p.name == "Settings":
            continue
        if p.name in array_names:
            layer_type = LAYER_TYPE_MAP[
                operation_spec.array_param_types.get(p.name, "image")
            ]
            p = p.replace(kind=inspect.Parameter.KEYWORD_ONLY, annotation=layer_type)
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

    sig_params: list = []
    if operation_spec.requires_layer:
        layer_type = LAYER_TYPE_MAP[operation_spec.layer_type]
        sig_params.append(
            inspect.Parameter(
                "layer", inspect.Parameter.KEYWORD_ONLY, annotation=layer_type
            )
        )
    sig_params += params_only + extra_params
    _dispatch.__doc__ = operation_spec.func.__doc__
    _dispatch.__signature__ = inspect.Signature(sig_params)

    extra_widget_options: dict[str, dict[str, str]] = {}
    if operation_spec.recordable_toggle:
        extra_widget_options["Record_in_Workflow"] = {
            "tooltip": "Uncheck to run this without saving it to the workflow."
        }
    if operation_spec.live_preview:
        extra_widget_options["Preview"] = {
            "tooltip": "Check to see a preview of the thresholding segmentation."
        }

    widget_layer_kwargs: dict = {}
    if operation_spec.requires_layer:
        widget_layer_kwargs["layer"] = {
            "label": "Layer",
            "tooltip": f"The {operation_spec.layer_type} layer to process.",
        }
    widget: FunctionGui = magicgui(
        _dispatch,
        call_button=f"Run {operation_spec.display_name}",
        **widget_layer_kwargs,
        **extra_widget_options,
        **operation_spec.widget_options,
    )
    widget_list: list[FunctionGui] = [widget]

    if operation_spec.allow_in_place:
        widget.In_Place.tooltip = "Warning: overwrites the selected layer's data "
        "directly. This is much faster and conserves memory, but cannot be undone."

    if operation_spec.live_preview:
        preview_layer_ref: dict[str, Layer | None] = {"layer": None}

        def _current_real_params() -> dict:
            real_names = [p.name for p in operation_spec.signature.parameters.values()][
                start_index:
            ]
            return {
                name: getattr(widget, name).value
                for name in real_names
                if hasattr(widget, name)
            }

        def _update_preview(event=None):
            if not widget.Preview.value or widget.layer.value is None:
                return
            if operation_spec.requires_layer:
                result = operation_spec.func(
                    widget.layer.value.data, **_current_real_params()
                )
            else:  # pragma: no cover
                result = operation_spec.func(**_current_real_params())
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

    if operation_spec.captures_viewer_settings:
        captured_settings: dict = {}

        def _capture_settings() -> None:
            target = widget.layer.value
            if target is None:
                return  # pragma: no cover

            canvas = viewer.window._qt_viewer.canvas.native
            raw: dict[str, Any] = {
                "blending": target.blending,
                "canvas_size": [canvas.width(), canvas.height()],
                "colormap": target.colormap,
                "camera_angles": list(viewer.scene.camera.angles),
                "camera_zoom": viewer.scene.camera.zoom,
                "camera_center": list(viewer.scene.camera.center),
                "camera_perspective": viewer.scene.camera.perspective,
                "camera_orientation": viewer.scene.camera.orientation,
                "depiction": target.depiction,
                "dims_ndisplay": viewer.dims.ndisplay,
                "opacity": target.opacity,
                "plane": target.plane,
                "projection_mode": target.projection_mode,
                "rendering": target.rendering,
            }
            if isinstance(target, Labels):
                raw.update(
                    {
                        "layer_type": "labels",
                        "iso_gradient_mode": target.iso_gradient_mode,
                    }
                )
            else:
                raw.update(
                    {
                        "layer_type": "image",
                        "attenuation": target.attenuation,
                        "contrast_limits": list(target.contrast_limits),
                        "gamma": target.gamma,
                        "interpolation2d": target.interpolation2d,
                        "interpolation3d": target.interpolation3d,
                        "iso_threshold": target.iso_threshold,
                    }
                )

            captured_settings.clear()
            captured_settings.update(to_jsonable(raw))

            if widget.call_button is not None:
                widget.call_button.text = "Settings Captured ✓"
            if widget.call_button is not None:
                widget.call_button.native.setEnabled(True)

        capture_widget = magicgui(
            _capture_settings, call_button="Capture View Settings"
        )
        widget_list.insert(0, capture_widget)

        if widget.call_button is not None:
            widget.call_button.native.setEnabled(False)

    for field_name, native_calls in operation_spec.native_options.items():
        field_widget = getattr(widget, field_name)
        for method_name, arg in native_calls.items():
            getattr(field_widget.native, method_name)(arg)

    if operation_spec.requires_layer:
        connect_layer_autoselect(viewer, widget, "layer", autoselect=True)
    for array_name in array_names:
        connect_layer_autoselect(viewer, widget, array_name, autoselect=False)

    container: CollapsibleContainer = CollapsibleContainer(
        widgets=widget_list, labels=False
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
