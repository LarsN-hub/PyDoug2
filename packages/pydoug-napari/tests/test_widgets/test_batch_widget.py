"""
Tests for PyDoug's napari plugin batch widget module.
"""

from pathlib import Path

from magicclass.widgets import CollapsibleContainer, ScrollableContainer
from magicgui.widgets import FunctionGui
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget

from pydoug_core.batch.recorder import Recorder
from pydoug_napari._widgets import batch_widget


def test_make_batch_widget(qtbot, tmp_path) -> None:

    record: Recorder = Recorder()
    record.record("test_name", {}, input_refs={"layer": "primary"})
    widget: QWidget = batch_widget.make_batch_widget(record)
    qtbot.addWidget(widget)
    container: ScrollableContainer = widget.container  # type: ignore
    save_widget: FunctionGui | None = container[0].get_widget("save_widget")
    clear_widget: FunctionGui | None = container[1].get_widget("clear_widget")
    save_widget.Parent_Folder.value = tmp_path  # type: ignore
    save_widget.Workflow_Folder_Name.value = "recording"  # type: ignore
    qtbot.mouseClick(save_widget.call_button.native, Qt.MouseButton.LeftButton)  # type: ignore
    assert Path(tmp_path / "recording").exists()
    qtbot.mouseClick(clear_widget.call_button.native, Qt.MouseButton.LeftButton)  # type: ignore
    assert len(record.steps) == 0


def _find_collapsible(
    container: ScrollableContainer, text: str
) -> CollapsibleContainer | None:
    for child in container:
        if isinstance(child, CollapsibleContainer) and child.text == text:
            return child
    return None


def test_clear_button_emits_steps_changed(qtbot):

    recorder: Recorder = Recorder()
    recorder.record("rotate", {"angle": 90}, input_refs={"layer": "primary"})
    received: list = []
    recorder.steps_changed.connect(lambda: received.append(True))
    widget: QWidget = batch_widget.make_batch_widget(recorder)
    qtbot.addWidget(widget)
    assert _find_collapsible(widget.container, "test") is None  # type: ignore
    clearcollapsible: CollapsibleContainer | None = _find_collapsible(
        widget.container,  # type: ignore
        "Clear Workflow",
    )
    assert clearcollapsible is not None
    clearwidget: FunctionGui | None = clearcollapsible.get_widget("clear_widget")
    assert clearwidget is not None
    assert clearwidget.call_button is not None
    qtbot.mouseClick(clearwidget.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: recorder.steps == [], timeout=5000)
    assert recorder.steps == []
    assert received
