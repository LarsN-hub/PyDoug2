"""
Tests for PyDoug's napari plugin main widget module.
"""

from pathlib import Path
from shutil import rmtree

from magicclass.widgets import CollapsibleContainer
from magicgui.widgets import FunctionGui
from napari.layers import Image
from napari.viewer import Viewer
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFileDialog, QMessageBox

from pydoug_napari._widgets import main_widget


def test_make_main_widget(
    make_napari_viewer, qtbot, monkeypatch, sample_3dgray_array, tmp_path
) -> None:

    def _return_tmp_path(*args, **kwargs):
        return tmp_path

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", _return_tmp_path)
    viewer: Viewer = make_napari_viewer()
    mainwidget: main_widget.MainWidget = main_widget.MainWidget(viewer)
    assert mainwidget.count() >= 1
    qtbot.addWidget(mainwidget)
    layer: Image | list[Image] = viewer.add_image(sample_3dgray_array, name="test")
    op_widget1: CollapsibleContainer = mainwidget.operation_widgets["mirror"]
    subwidget1: FunctionGui | None = op_widget1.get_widget("_dispatch")
    op_widget2: CollapsibleContainer = mainwidget.operation_widgets["stats"]
    subwidget2: FunctionGui | None = op_widget2.get_widget("_dispatch")
    assert subwidget1 is not None
    assert subwidget2 is not None
    subwidget1.layer.value = layer
    assert subwidget1.call_button is not None
    qtbot.mouseClick(subwidget1.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(viewer.layers) == 2, timeout=5000)
    assert len(mainwidget.recorder.steps) == 1
    assert len(viewer.layers) == 2
    qtbot.mouseClick(subwidget1.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(viewer.layers) == 3, timeout=5000)
    assert len(mainwidget.recorder.steps) == 2
    assert len(viewer.layers) == 3
    viewer.layers.remove(viewer.layers[-1])
    assert len(mainwidget.recorder.steps) == 1
    assert subwidget1.layer != layer
    subwidget2.layer.value = layer
    assert subwidget2.Record_in_Workflow is not None
    subwidget2.Record_in_Workflow.value = True
    assert subwidget2.call_button is not None
    qtbot.mouseClick(subwidget2.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(mainwidget.recorder.steps) == 2, timeout=5000)
    assert len(mainwidget.recorder.steps) == 2
    assert mainwidget.results_panel.export_all_button is not None
    qtbot.mouseClick(
        mainwidget.results_panel.export_all_button, Qt.MouseButton.LeftButton
    )
    qtbot.waitUntil(lambda: Path(tmp_path / "Stats/values.csv").exists(), timeout=5000)
    assert Path(tmp_path / "Stats/values.csv").exists()
    rmtree(str(Path(tmp_path / "Stats")))
    assert not Path(tmp_path / "Stats/values.csv").exists()
    mainwidget.results_panel.log_widget.setCurrentRow(0)  # type: ignore
    qtbot.mouseClick(
        mainwidget.results_panel.export_selected_button, Qt.MouseButton.LeftButton
    )
    qtbot.waitUntil(lambda: Path(tmp_path / "Stats/values.csv").exists(), timeout=5000)
    assert Path(tmp_path / "Stats/values.csv").exists()


def test_cascade_delete_removes_dependent_steps(
    make_napari_viewer, qtbot, sample_3dgray_array, registered_half, monkeypatch
):
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)  # type: ignore
    viewer: Viewer = make_napari_viewer()
    widget: main_widget.MainWidget = main_widget.MainWidget(viewer)
    qtbot.addWidget(widget)
    layer: Image | list[Image] = viewer.add_image(sample_3dgray_array, name="test")
    op_widget: CollapsibleContainer = widget.operation_widgets["_half"]
    subwidget: FunctionGui | None = op_widget.get_widget("_dispatch")
    assert subwidget is not None
    subwidget.layer.value = layer
    assert subwidget.call_button is not None
    qtbot.mouseClick(subwidget.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(widget.recorder.steps) == 1, timeout=5000)
    qtbot.mouseClick(subwidget.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(widget.recorder.steps) == 2, timeout=5000)
    qtbot.mouseClick(subwidget.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(widget.recorder.steps) == 3, timeout=5000)
    viewer.add_image(sample_3dgray_array, name="test2")
    viewer.layers.remove(viewer.layers[-1])
    viewer.layers.remove(viewer.layers[1])
    assert len(widget.recorder.steps) == 0
    assert len(viewer.layers) == 1
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)  # type: ignore
    qtbot.mouseClick(subwidget.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(widget.recorder.steps) == 1, timeout=5000)
    qtbot.mouseClick(subwidget.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(widget.recorder.steps) == 2, timeout=5000)
    qtbot.mouseClick(subwidget.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(widget.recorder.steps) == 3, timeout=5000)
    viewer.layers.remove(viewer.layers[1])
    assert len(widget.recorder.steps) == 2
    assert len(viewer.layers) == 3
