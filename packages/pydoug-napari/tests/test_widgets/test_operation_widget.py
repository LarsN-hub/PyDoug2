"""
Tests for PyDoug's napari plugin operation_widget module.
"""

import json
from time import sleep

import numpy as np
import pytest
from magicclass.widgets import CollapsibleContainer
from magicgui.widgets import FunctionGui
from napari.layers import Image, Labels, Layer
from napari.viewer import Viewer
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QMainWindow
from skimage.measure import marching_cubes
from skimage.morphology import ball

from pydoug_core.batch.recorder import Recorder
from pydoug_core.registry import OperationSpec, get_operation
from pydoug_napari._widgets.operation_widget import make_operation_widget
from pydoug_napari._widgets.results_log_widget import ResultsLog


def _half(imarray: np.ndarray, factor: float = 0.5) -> np.ndarray:

    output: np.ndarray = np.round(imarray * factor).astype(
        imarray.dtype
    )  # pragma: no cover
    sleep(1)  # pragma: no cover
    return output  # pragma: no cover
    # covered in a separate Qt threading event


@pytest.fixture
def half_spec() -> OperationSpec:
    """
    Fixture to create an operation for the dummy function "_half".
    """

    return OperationSpec(
        name="_half",
        func=_half,
        category="testing",
        display_name="Half",
    )


def test_operation_widget_runs_and_records(
    make_napari_viewer, half_spec, qtbot, sample_3dgray_array
) -> None:

    viewer: Viewer = make_napari_viewer()
    recorder: Recorder = Recorder()
    results_log: ResultsLog = ResultsLog()
    layer: Image | list[Image] = viewer.add_image(
        sample_3dgray_array, name="test_layer"
    )
    widget: CollapsibleContainer = make_operation_widget(
        half_spec, viewer, recorder, results_log
    )
    qtbot.addWidget(widget.native)
    subwidget: FunctionGui | None = widget.get_widget("_dispatch")
    assert subwidget is not None
    subwidget.layer.value = layer
    subwidget.factor.value = 0.5
    assert subwidget.call_button is not None
    qtbot.mouseClick(subwidget.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(viewer.layers) == 2, timeout=5000)
    output_layer: Layer = viewer.layers[-1]
    np.testing.assert_array_equal(
        output_layer.data,
        np.round(sample_3dgray_array * 0.5).astype(sample_3dgray_array.dtype),
    )
    assert len(recorder.steps) == 1
    assert recorder.steps[0].operation_name == "_half"
    assert recorder.steps[0].params == {"factor": 0.5}


def test_new_layer_becomes_selected(make_napari_viewer, half_spec) -> None:

    viewer: Viewer = make_napari_viewer()
    layer1: Image | list[Image] = viewer.add_image(np.zeros((5, 5)), name="first")
    widget: CollapsibleContainer = make_operation_widget(
        half_spec, viewer, Recorder(), ResultsLog()
    )
    subwidget: FunctionGui | None = widget.get_widget("_dispatch")
    assert subwidget is not None
    subwidget.layer.value = layer1
    layer2: Image | list[Image] = viewer.add_image(np.ones((5, 5)), name="second")
    assert subwidget.layer.value is layer2


def test_widget_double_click(
    make_napari_viewer, half_spec, qtbot, sample_3dgray_array
) -> None:

    viewer: Viewer = make_napari_viewer()
    recorder: Recorder = Recorder()
    results_log: ResultsLog = ResultsLog()
    layer: Image | list[Image] = viewer.add_image(
        sample_3dgray_array, name="test_layer"
    )
    widget: CollapsibleContainer = make_operation_widget(
        half_spec, viewer, recorder, results_log
    )
    qtbot.addWidget(widget.native)
    subwidget: FunctionGui | None = widget.get_widget("_dispatch")
    assert subwidget is not None
    subwidget.layer.value = layer
    subwidget.factor.value = 0.5
    assert subwidget.call_button is not None
    qtbot.mouseClick(subwidget.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(subwidget.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(viewer.layers) == 2, timeout=5000)
    assert len(viewer.layers) == 2


def test_on_error_reraises_and_resets_button(
    make_napari_viewer, qtbot, sample_3dgray_array, registered_half, monkeypatch
):

    viewer: Viewer = make_napari_viewer()
    recorder: Recorder = Recorder()
    results_log: ResultsLog = ResultsLog()
    layer: Image | list[Image] = viewer.add_image(
        sample_3dgray_array, name="test_layer"
    )
    spec: OperationSpec = get_operation(registered_half)
    container: CollapsibleContainer = make_operation_widget(
        spec, viewer, recorder, results_log
    )
    qtbot.addWidget(container.native)
    subwidget: FunctionGui | None = container.get_widget("_dispatch")
    assert subwidget is not None
    subwidget.layer.value = layer
    assert subwidget.call_button is not None

    def _boom(imarray, **kwargs):  # pragma: no cover
        raise ValueError("simulated failure")  # pragma: no cover

    monkeypatch.setattr(spec, "func", _boom)
    with qtbot.captureExceptions() as exceptions:
        qtbot.mouseClick(subwidget.call_button.native, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: len(exceptions) > 0, timeout=5000)
    value_errors = [e for e in exceptions if e[0] is ValueError]
    assert len(value_errors) == 1
    assert "simulated failure" in str(value_errors[0][1])
    assert subwidget.call_button.enabled
    assert subwidget.call_button.text == f"Run {spec.display_name}"


def test_layer_refs(
    make_napari_viewer, qtbot, sample_3dgray_array, registered_half
) -> None:

    viewer: Viewer = make_napari_viewer()
    record: Recorder = Recorder()
    results_log: ResultsLog = ResultsLog()
    layer: Image | list[Image] = viewer.add_image(
        sample_3dgray_array, name="test_layer"
    )
    spec: OperationSpec = get_operation(registered_half)
    container: CollapsibleContainer = make_operation_widget(
        spec, viewer, record, results_log
    )
    qtbot.addWidget(container.native)
    subwidget: FunctionGui | None = container.get_widget("_dispatch")
    assert subwidget is not None
    assert subwidget.layer.value == layer
    assert subwidget.call_button is not None
    qtbot.mouseClick(subwidget.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(viewer.layers) == 2, timeout=5000)
    assert len(viewer.layers) == 2
    assert record.steps[0].input_refs["layer"] == "primary"
    assert subwidget.layer.value != layer
    subwidget.layer.value = layer
    qtbot.mouseClick(subwidget.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(viewer.layers) == 3, timeout=5000)
    assert len(viewer.layers) == 3
    assert record.steps[1].input_refs["layer"] == "primary"
    qtbot.mouseClick(subwidget.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(viewer.layers) == 4, timeout=5000)
    assert len(viewer.layers) == 4
    assert record.steps[2].input_refs["layer"] != "primary"
    assert record.steps[2].input_refs["layer"] == f"step:{record.steps[1].step_id}"


def test_array_parameter_behavior(
    make_napari_viewer,
    qtbot,
    sample_3dgray_array,
    sample_masks,
    registered_mask,
    registered_bool,
) -> None:

    viewer: Viewer = make_napari_viewer()
    record: Recorder = Recorder()
    results_log: ResultsLog = ResultsLog()
    imlayer: Image | list[Image] = viewer.add_image(
        sample_3dgray_array, name="test_layer"
    )
    masklayer: Image | list[Image] = viewer.add_image(
        sample_masks["3dgray"], name="test_mask"
    )
    spec1: OperationSpec = get_operation(registered_mask)
    spec2: OperationSpec = get_operation(registered_bool)
    assert spec1.array_params is not None
    container1: CollapsibleContainer = make_operation_widget(
        spec1, viewer, record, results_log
    )
    container2: CollapsibleContainer = make_operation_widget(
        spec2, viewer, record, results_log
    )
    qtbot.addWidget(container1.native)
    qtbot.addWidget(container2.native)
    subwidget1: FunctionGui | None = container1.get_widget("_dispatch")
    subwidget2: FunctionGui | None = container2.get_widget("_dispatch")
    assert subwidget1 is not None
    assert subwidget1.layer.value == imlayer
    assert subwidget1.call_button is not None
    assert not subwidget1.In_Place.value
    assert subwidget2 is not None
    assert subwidget2.layer.value == imlayer
    assert subwidget2.call_button is not None
    subwidget1.maskarray.value = masklayer
    subwidget1.Apply_Mask.value = True
    qtbot.mouseClick(subwidget1.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(viewer.layers) == 3, timeout=5000)
    assert len(viewer.layers) == 3
    subwidget2.layer.value = viewer.layers[-1]
    qtbot.mouseClick(subwidget2.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(viewer.layers) == 4, timeout=5000)
    assert len(viewer.layers) == 4
    subwidget1.layer.value = viewer.layers[0]
    subwidget1.maskarray.value = viewer.layers[-1]
    qtbot.mouseClick(subwidget1.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(viewer.layers) == 5, timeout=5000)
    assert len(viewer.layers) == 5
    subwidget1.Apply_Mask.value = False
    subwidget1.layer.value = imlayer
    subwidget1.maskarray.value = viewer.layers[-2]
    qtbot.mouseClick(subwidget1.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(viewer.layers) == 6, timeout=5000)
    assert np.all(viewer.layers[0].data == viewer.layers[-1].data)


def test_in_place_behavior(
    make_napari_viewer,
    qtbot,
    sample_3dgray_array,
    sample_masks,
    registered_mask,
) -> None:

    viewer: Viewer = make_napari_viewer()
    record: Recorder = Recorder()
    results_log: ResultsLog = ResultsLog()
    imlayer: Image | list[Image] = viewer.add_image(
        sample_3dgray_array, name="test_layer"
    )
    masklayer: Image | list[Image] = viewer.add_image(
        sample_masks["3dgray"], name="test_mask"
    )
    spec: OperationSpec = get_operation(registered_mask)
    assert spec.array_params is not None
    container: CollapsibleContainer = make_operation_widget(
        spec, viewer, record, results_log
    )
    qtbot.addWidget(container.native)
    subwidget: FunctionGui | None = container.get_widget("_dispatch")
    assert subwidget is not None
    assert subwidget.call_button is not None
    subwidget.layer.value = imlayer
    subwidget.maskarray.value = masklayer
    subwidget.Apply_Mask.value = True
    subwidget.In_Place.value = False
    qtbot.mouseClick(subwidget.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(viewer.layers) == 3, timeout=5000)
    assert len(viewer.layers) == 3
    subwidget.layer.value = imlayer
    subwidget.maskarray.value = masklayer
    subwidget.In_Place.value = True
    qtbot.mouseClick(subwidget.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(record.steps) == 2, timeout=5000)
    assert len(record.steps) == 2
    assert len(viewer.layers) == 3
    assert np.all(viewer.layers[0].data == viewer.layers[-1].data)


def test_non_image_output(
    make_napari_viewer,
    qtbot,
    sample_3dgray_array,
    registered_calc,
    registered_calc_vr,
    registered_plot,
    monkeypatch,
) -> None:

    viewer: Viewer = make_napari_viewer()
    record: Recorder = Recorder()
    results_log: ResultsLog = ResultsLog()
    imlayer: Image | list[Image] = viewer.add_image(
        sample_3dgray_array, name="test_layer"
    )
    spec1: OperationSpec = get_operation(registered_calc)
    container1: CollapsibleContainer = make_operation_widget(
        spec1, viewer, record, results_log
    )
    qtbot.addWidget(container1.native)
    subwidget1: FunctionGui | None = container1.get_widget("_dispatch")
    spec2: OperationSpec = get_operation(registered_plot)
    container2: CollapsibleContainer = make_operation_widget(
        spec2, viewer, record, results_log
    )
    qtbot.addWidget(container2.native)
    spec3: OperationSpec = get_operation(registered_calc_vr)
    container3: CollapsibleContainer = make_operation_widget(
        spec3, viewer, record, results_log
    )
    qtbot.addWidget(container3.native)
    subwidget1: FunctionGui | None = container1.get_widget("_dispatch")
    subwidget2: FunctionGui | None = container2.get_widget("_dispatch")
    subwidget3: FunctionGui | None = container3.get_widget("_dispatch")
    assert subwidget1 is not None
    assert subwidget1.layer.value == imlayer
    assert subwidget1.call_button is not None
    assert subwidget2 is not None
    assert subwidget2.layer.value == imlayer
    assert subwidget2.call_button is not None
    assert subwidget3 is not None
    assert subwidget3.layer.value == imlayer
    assert subwidget3.call_button is not None
    qtbot.mouseClick(subwidget1.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(record.steps) == 1, timeout=5000)
    assert len(record.steps) == 1
    qtbot.mouseClick(subwidget3.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(record.steps) == 2, timeout=5000)
    assert len(record.steps) == 2
    monkeypatch.setattr(QMainWindow, "show", lambda *a, **k: None)
    qtbot.mouseClick(subwidget2.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(record.steps) == 3, timeout=5000)
    assert len(record.steps) == 3


def test_live_preview(
    make_napari_viewer,
    qtbot,
    sample_3dgray_array,
    registered_threshold,
) -> None:

    viewer: Viewer = make_napari_viewer()
    record: Recorder = Recorder()
    results_log: ResultsLog = ResultsLog()
    imlayer: Image | list[Image] = viewer.add_image(
        sample_3dgray_array, name="test_layer"
    )
    spec: OperationSpec = get_operation(registered_threshold)
    container: CollapsibleContainer = make_operation_widget(
        spec, viewer, record, results_log
    )
    qtbot.addWidget(container.native)
    subwidget: FunctionGui | None = container.get_widget("_dispatch")
    assert subwidget is not None
    assert subwidget.call_button is not None
    assert subwidget.Preview is not None
    subwidget.layer.value = imlayer
    subwidget.Preview.value = True
    assert len(viewer.layers) == 2
    subwidget.Preview.value = False
    assert len(viewer.layers) == 1
    subwidget.Preview.value = True
    assert len(viewer.layers) == 2
    qtbot.mouseClick(subwidget.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(record.steps) == 1, timeout=5000)
    assert len(record.steps) == 1
    assert not subwidget.Preview.value
    assert len(viewer.layers) == 2


def test_preview_never_hijacks_layer_selection(
    make_napari_viewer, qtbot, sample_2dgray_array, registered_threshold
) -> None:

    viewer: Viewer = make_napari_viewer()
    layer: Image | list[Image] = viewer.add_image(sample_2dgray_array, name="test")
    spec: OperationSpec = get_operation(registered_threshold)
    container: CollapsibleContainer = make_operation_widget(
        spec, viewer, Recorder(), ResultsLog()
    )
    qtbot.addWidget(container.native)
    subwidget: FunctionGui | None = container.get_widget("_dispatch")
    assert subwidget is not None
    subwidget.layer.value = layer
    subwidget.Preview.value = True
    assert subwidget.layer.value is layer
    assert len(viewer.layers) == 2


def test_select_last_respects_field_type(
    make_napari_viewer, qtbot, sample_2dgray_array, registered_label_count
) -> None:

    viewer: Viewer = make_napari_viewer()
    image_layer: Image | list[Image] = viewer.add_image(
        sample_2dgray_array, name="image"
    )
    labels_layer: Labels | list[Labels] = viewer.add_labels(  # type: ignore
        np.zeros(sample_2dgray_array.shape, dtype=int), name="labels"
    )
    spec: OperationSpec = get_operation(registered_label_count)
    container: CollapsibleContainer = make_operation_widget(
        spec, viewer, Recorder(), ResultsLog()
    )
    qtbot.addWidget(container.native)
    subwidget: FunctionGui | None = container.get_widget("_dispatch")
    assert subwidget is not None
    subwidget.layer.value = labels_layer
    viewer.layers.remove(image_layer)  # type: ignore
    assert subwidget.layer.value is labels_layer


def test_captures_viewer_settings(
    make_napari_viewer,
    qtbot,
    sample_2dgray_array,
    registered_capture_screen_image,
    registered_capture_screen_labels,
) -> None:

    viewer: Viewer = make_napari_viewer()
    imlayer: Image | list[Image] = viewer.add_image(sample_2dgray_array, name="test")
    lablayer: Image | list[Image] = viewer.add_labels(  # type: ignore
        sample_2dgray_array.astype(np.int32), name="test"
    )
    spec1: OperationSpec = get_operation(registered_capture_screen_image)
    spec2: OperationSpec = get_operation(registered_capture_screen_labels)
    recorder: Recorder = Recorder()
    container1: CollapsibleContainer = make_operation_widget(
        spec1, viewer, recorder, ResultsLog()
    )
    container2: CollapsibleContainer = make_operation_widget(
        spec2, viewer, recorder, ResultsLog()
    )
    qtbot.addWidget(container1.native)

    capture_widget1: FunctionGui | None = container1.get_widget("_capture_settings")
    run_widget1: FunctionGui | None = container1.get_widget("_dispatch")
    assert capture_widget1 is not None and run_widget1 is not None
    assert (
        capture_widget1.call_button is not None and run_widget1.call_button is not None
    )
    run_widget1.layer.value = imlayer
    assert not run_widget1.call_button.native.isEnabled()
    qtbot.mouseClick(capture_widget1.call_button.native, Qt.MouseButton.LeftButton)
    assert run_widget1.call_button.native.isEnabled()
    qtbot.mouseClick(run_widget1.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(viewer.layers) == 3, timeout=5000)
    assert len(recorder.steps) == 1
    json.dumps(recorder.steps[0].params)

    capture_widget2: FunctionGui | None = container2.get_widget("_capture_settings")
    run_widget2: FunctionGui | None = container2.get_widget("_dispatch")
    assert capture_widget2 is not None and run_widget2 is not None
    assert (
        capture_widget2.call_button is not None and run_widget2.call_button is not None
    )
    run_widget2.layer.value = lablayer
    assert not run_widget2.call_button.native.isEnabled()
    qtbot.mouseClick(capture_widget2.call_button.native, Qt.MouseButton.LeftButton)
    assert run_widget2.call_button.native.isEnabled()
    qtbot.mouseClick(run_widget2.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(viewer.layers) == 4, timeout=5000)
    assert len(recorder.steps) == 2
    json.dumps(recorder.steps[1].params)


def test_no_layer_behavior(
    make_napari_viewer, qtbot, registered_generator, registered_half
) -> None:

    viewer: Viewer = make_napari_viewer()
    recorder: Recorder = Recorder()
    results_log: ResultsLog = ResultsLog()
    spec1: OperationSpec = get_operation(registered_generator)
    spec2: OperationSpec = get_operation(registered_half)
    widget1: CollapsibleContainer = make_operation_widget(
        spec1, viewer, recorder, results_log
    )
    widget2: CollapsibleContainer = make_operation_widget(
        spec2, viewer, recorder, results_log
    )
    qtbot.addWidget(widget1.native)
    qtbot.addWidget(widget2.native)
    subwidget1: FunctionGui | None = widget1.get_widget("_dispatch")
    subwidget2: FunctionGui | None = widget2.get_widget("_dispatch")
    assert subwidget1 is not None
    assert subwidget2 is not None
    assert len(viewer.layers) == 0
    qtbot.mouseClick(subwidget1.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(viewer.layers) == 1, timeout=5000)
    assert len(viewer.layers) == 1
    subwidget2.layer.value = viewer.layers[-1]
    qtbot.mouseClick(subwidget2.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(viewer.layers) == 2, timeout=5000)
    assert len(viewer.layers) == 2


def test_surface_behavior(
    make_napari_viewer, qtbot, registered_surface, registered_surface_calc
) -> None:

    viewer: Viewer = make_napari_viewer()
    recorder: Recorder = Recorder()
    results_log: ResultsLog = ResultsLog()
    spec1: OperationSpec = get_operation(registered_surface)
    spec2: OperationSpec = get_operation(registered_surface_calc)
    widget1: CollapsibleContainer = make_operation_widget(
        spec1, viewer, recorder, results_log
    )
    widget2: CollapsibleContainer = make_operation_widget(
        spec2, viewer, recorder, results_log
    )
    qtbot.addWidget(widget1.native)
    qtbot.addWidget(widget2.native)
    subwidget1: FunctionGui | None = widget1.get_widget("_dispatch")
    subwidget2: FunctionGui | None = widget2.get_widget("_dispatch")
    assert subwidget1 is not None
    assert subwidget2 is not None
    assert len(viewer.layers) == 0
    qtbot.mouseClick(subwidget1.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(viewer.layers) == 1, timeout=5000)
    assert len(viewer.layers) == 1
    subwidget2.layer.value = viewer.layers[-1]
    qtbot.mouseClick(subwidget2.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(recorder.steps) == 2, timeout=5000)
    assert len(recorder.steps) == 2
    array: np.ndarray = ball(25)  # type: ignore
    verts, faces, _, values = marching_cubes(array)
    viewer.add_surface((verts, faces, values))  # type: ignore
    subwidget2.layer.value = viewer.layers[-1]
    qtbot.mouseClick(subwidget2.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(recorder.steps) == 3, timeout=5000)
    assert len(recorder.steps) == 3


def test_shapes_behavior(
    make_napari_viewer,
    qtbot,
    registered_shapes,
    registered_shapes_primary,
) -> None:

    viewer: Viewer = make_napari_viewer()
    recorder: Recorder = Recorder()
    results_log: ResultsLog = ResultsLog()
    spec1: OperationSpec = get_operation(registered_shapes)
    spec2: OperationSpec = get_operation(registered_shapes_primary)
    widget1: CollapsibleContainer = make_operation_widget(
        spec1, viewer, recorder, results_log
    )
    widget2: CollapsibleContainer = make_operation_widget(
        spec2, viewer, recorder, results_log
    )
    qtbot.addWidget(widget1.native)
    qtbot.addWidget(widget2.native)
    subwidget1: FunctionGui | None = widget1.get_widget("_dispatch")
    subwidget2: FunctionGui | None = widget2.get_widget("_dispatch")
    assert subwidget1 is not None
    assert subwidget2 is not None
    viewer.add_shapes(  # type: ignore
        np.array([[3, 3], [7, 3], [7, 7], [7, 3]]), shape_type="rectangle"
    )
    assert len(viewer.layers) == 1
    subwidget1.layer.value = viewer.layers[-1]
    qtbot.mouseClick(subwidget1.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(viewer.layers) == 2, timeout=5000)
    assert len(viewer.layers) == 2
    subwidget2.layer.value = viewer.layers[0]
    subwidget2.imarray.value = viewer.layers[-1]
    assert len(recorder.steps) == 0
    qtbot.mouseClick(subwidget2.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(recorder.steps) == 1, timeout=5000)
    assert len(recorder.steps) == 1


def test_VisualResult_behavior() -> None:

    pass
