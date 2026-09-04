"""
Tests for PyDoug's napari plugin batch widget module.
"""

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget

from pydoug_core.batch.recorder import Recorder
from pydoug_napari._widgets import batch_widget


def test_make_batch_widget(qtbot, tmp_path) -> None:

    record: Recorder = Recorder()
    record.record("mirror", {})
    widget: QWidget = batch_widget.make_batch_widget(record)
    qtbot.addWidget(widget)
    assert widget.save_widget.call_button is not None  # type: ignore[attr-defined]
    assert widget.load_widget.call_button is not None  # type: ignore[attr-defined]
    assert widget.clear_widget.call_button is not None  # type: ignore[attr-defined]
    widget.save_widget.File_Name.value = tmp_path / "recording.json"  # type: ignore[attr-defined]
    qtbot.mouseClick(widget.save_widget.call_button.native, Qt.MouseButton.LeftButton)  # type: ignore[attr-defined]
    assert Path(tmp_path / "recording.json").exists()
    qtbot.mouseClick(widget.clear_widget.call_button.native, Qt.MouseButton.LeftButton)  # type: ignore[attr-defined]
    assert len(record.steps) == 0
    widget.load_widget.File_Path.value = tmp_path / "recording.json"  # type: ignore[attr-defined]
    qtbot.mouseClick(widget.load_widget.call_button.native, Qt.MouseButton.LeftButton)  # type: ignore[attr-defined]
    assert len(record.steps) == 1
