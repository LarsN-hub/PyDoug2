"""
Tests for PyDoug's napari plugin main widget module.
"""

from magicclass.widgets import CollapsibleContainer
from magicgui.widgets import FunctionGui
from napari.layers import Image
from napari.viewer import Viewer
from qtpy.QtCore import Qt

from pydoug_napari._widgets import main_widget


def test_make_main_widget(make_napari_viewer, qtbot, sample_3dgray_array) -> None:

    viewer: Viewer = make_napari_viewer()
    mainwidget: main_widget.MainWidget = main_widget.MainWidget(viewer)
    assert mainwidget.count() >= 1
    qtbot.addWidget(mainwidget)
    layer: Image | list[Image] = viewer.add_image(sample_3dgray_array, name="test")
    op_widget: CollapsibleContainer = mainwidget.operation_widgets["mirror"]
    subwidget: FunctionGui | None = op_widget.get_widget("_dispatch")
    assert subwidget is not None
    subwidget.layer.value = layer
    assert subwidget.call_button is not None
    qtbot.mouseClick(subwidget.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(mainwidget.recorder.steps) == 1, timeout=5000)
    assert len(mainwidget.recorder.steps) == 1
    qtbot.mouseClick(subwidget.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(mainwidget.recorder.steps) == 2, timeout=5000)
    assert len(mainwidget.recorder.steps) == 2
    viewer.layers.remove(viewer.layers[-1])
    assert len(mainwidget.recorder.steps) == 1
    assert subwidget.layer != layer
