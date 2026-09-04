# packages/proj-napari/tests/test_widgets/test_operation_widget.py
"""
Tests for PyDoug's napari plugin operation_widget module.
"""

import numpy as np
import pytest
from magicgui.widgets import FunctionGui
from napari.layers import Image, Layer
from napari.viewer import Viewer
from qtpy.QtCore import Qt

from pydoug_core.batch.recorder import Recorder
from pydoug_core.registry import OperationSpec
from pydoug_napari._widgets.operation_widget import make_operation_widget


def _half(imarray: np.ndarray, factor: float = 0.5) -> np.ndarray:

    return np.round(imarray * factor).astype(imarray.dtype)


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
    layer: Image | list[Image] = viewer.add_image(
        sample_3dgray_array, name="test_layer"
    )
    widget: FunctionGui = make_operation_widget(half_spec, viewer, recorder)
    qtbot.addWidget(widget.native)
    widget.layer.value = layer
    widget.factor.value = 0.5
    assert widget.call_button is not None
    qtbot.mouseClick(widget.call_button.native, Qt.MouseButton.LeftButton)
    assert len(viewer.layers) == 2
    output_layer: Layer = viewer.layers[-1]
    np.testing.assert_array_equal(
        output_layer.data,
        np.round(sample_3dgray_array * 0.5).astype(sample_3dgray_array.dtype),
    )
    assert len(recorder.steps) == 1
    assert recorder.steps[0].operation_name == "_half"
    assert recorder.steps[0].params == {"factor": 0.5}
