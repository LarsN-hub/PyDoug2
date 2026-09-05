"""
Tests for PyDoug's napari plugin batch widget module.
"""

from pathlib import Path

from magicclass.widgets import CollapsibleContainer, ScrollableContainer
from magicgui.widgets import FunctionGui
from qtpy.QtCore import Qt

from pydoug_core.batch.recorder import Recorder
from pydoug_napari._widgets import batch_widget


def test_make_batch_widget(qtbot, tmp_path) -> None:

    record: Recorder = Recorder()
    record.record("mirror", {})
    scroll_widget: ScrollableContainer = batch_widget.make_batch_widget(record)
    qtbot.addWidget(scroll_widget.native)
    save_widget: FunctionGui | None = scroll_widget[0].get_widget("save_widget")
    load_widget: FunctionGui | None = scroll_widget[1].get_widget("load_widget")
    clear_widget: FunctionGui | None = scroll_widget[2].get_widget("clear_widget")
    save_widget.File_Name.value = tmp_path / "recording.json"  # type: ignore[attr-defined]
    qtbot.mouseClick(save_widget.call_button.native, Qt.MouseButton.LeftButton)  # type: ignore[attr-defined]
    assert Path(tmp_path / "recording.json").exists()
    qtbot.mouseClick(clear_widget.call_button.native, Qt.MouseButton.LeftButton)  # type: ignore[attr-defined]
    assert len(record.steps) == 0
    load_widget.File_Path.value = tmp_path / "recording.json"  # type: ignore[attr-defined]
    qtbot.mouseClick(load_widget.call_button.native, Qt.MouseButton.LeftButton)  # type: ignore[attr-defined]
    assert len(record.steps) == 1


def _find_collapsible(
    container: ScrollableContainer, text: str
) -> CollapsibleContainer | None:
    for child in container:
        if isinstance(child, CollapsibleContainer) and child.text == text:
            return child
    return None


def test_clear_button_emits_steps_changed(qtbot):

    recorder: Recorder = Recorder()
    recorder.record("rotate", {"angle": 90})
    received: list = []
    recorder.steps_changed.connect(lambda: received.append(True))
    widget: ScrollableContainer = batch_widget.make_batch_widget(recorder)
    qtbot.addWidget(widget.native)
    assert _find_collapsible(widget, "test") is None
    clearcollapsible: CollapsibleContainer | None = _find_collapsible(
        widget, "Clear Workflow"
    )
    assert clearcollapsible is not None
    clearwidget: FunctionGui | None = clearcollapsible.get_widget("clear_widget")
    assert clearwidget is not None
    assert clearwidget.call_button is not None
    qtbot.mouseClick(clearwidget.call_button.native, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: recorder.steps == [], timeout=5000)
    assert recorder.steps == []
    assert received
