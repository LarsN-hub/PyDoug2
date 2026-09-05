"""
Tests for PyDoug's napari plugin batch widget module.
"""

from pathlib import Path

from magicclass.widgets import ScrollableContainer
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
